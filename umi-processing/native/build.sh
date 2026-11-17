#!/usr/bin/env bash
# Build UMI's ORB-SLAM3 fork (github.com/cheng-chi/ORB_SLAM3 at b741dca) with umiproc's patches, and the umiproc_slam
# front-end. Laptop (Ubuntu 22.04 x86_64) and Jetson Orin Nano (JetPack, aarch64). Idempotent: a second run only
# rebuilds what changed. See native/README.md.
#
#   umi-processing/native/build.sh
#
# Environment:
#   ORB_SLAM3_UMI_DIR  where the fork lives (default ~/ORB_SLAM3_umi); an existing checkout must be at b741dca
#   JOBS               parallel compile jobs (default: 2 below 10 GB of RAM, else RAM/2 GB capped at the cores)
#   UMIPROC_OPENCV     auto (default) | system | build: which OpenCV the fork is built against. The fork needs
#                      OpenCV's contrib aruco module with the pre-4.7 API (DetectorParameters::create(),
#                      Ptr<Dictionary> getPredefinedDictionary). auto uses the system OpenCV when it qualifies, else
#                      builds a private OpenCV 4.5.4 + contrib (the laptop's apt version) under the fork's Thirdparty/.
#   OpenCV_DIR         with system: the OpenCVConfig.cmake folder to use instead of the default one
#   PATCHES_UPTO       e.g. 0006: build a variant with only the patches up to that number, for A/B runs. It goes to
#                      ${ORB_SLAM3_UMI_DIR}/umiproc_build_0006/ with its own copy of the fork's sources (a git
#                      worktree in fork/), its own libORB_SLAM3.so and its own umiproc_slam; the main build is not
#                      touched (third-party libraries, OpenCV and the vocabulary are shared). Remove it with
#                      rm -rf ${ORB_SLAM3_UMI_DIR}/umiproc_build_0006 && git -C ${ORB_SLAM3_UMI_DIR} worktree prune
#
# Results: ${ORB_SLAM3_UMI_DIR}/umiproc_build/umiproc_slam, ${ORB_SLAM3_UMI_DIR}/Vocabulary/ORBvoc.txt,
# ${ORB_SLAM3_UMI_DIR}/umiproc_build/BUILD_INFO; logs in ${ORB_SLAM3_UMI_DIR}/umiproc_build/logs/
# (umiproc_build_NNNN/ for a PATCHES_UPTO variant).
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_DIR="${ORB_SLAM3_UMI_DIR:-$HOME/ORB_SLAM3_umi}"
FORK_URL="https://github.com/cheng-chi/ORB_SLAM3.git"
FORK_COMMIT="b741dca39015330ef4bcc3a85f89493503ade04b"
PANGOLIN_PATH="Thirdparty/Pangolin"
PANGOLIN_URL="https://github.com/stevenlovegrove/Pangolin.git"       # the fork's .gitmodules has an ssh URL
PANGOLIN_COMMIT="d484494645cb7361374ac0ef6b27e9ee6feffbd7"
OPENCV_VERSION="4.5.4"
OPENCV_COMMIT="4223495e6cd67011f86b8ecd9be1fa105018f3b1"              # tag 4.5.4
OPENCV_CONTRIB_COMMIT="2bc3e1a169afd2675ef68221176f32cc0ea5427a"      # tag 4.5.4
OPENCV_MODULES="core,imgproc,imgcodecs,highgui,features2d,flann,calib3d,aruco"   # what the fork uses
UMIPROC_OPENCV="${UMIPROC_OPENCV:-auto}"
PATCH_DIR="$HERE/patches"
FRONTEND_SRC="$HERE/umi_slam"
PATCHES_UPTO="${PATCHES_UPTO:-}"
OUT_DIR="$FORK_DIR/umiproc_build${PATCHES_UPTO:+_$PATCHES_UPTO}"
LOG_DIR="$OUT_DIR/logs"

say() { printf '== %s\n' "$*"; }
die() { printf '\nbuild.sh: %s\n' "$*" >&2; exit 1; }

CURRENT_LOG=""
on_error() {
    local rc=$?
    if [[ -n "$CURRENT_LOG" && -f "$CURRENT_LOG" ]]; then
        printf '\nbuild.sh: failed (exit %s); last lines of %s:\n' "$rc" "$CURRENT_LOG" >&2
        tail -n 40 "$CURRENT_LOG" >&2
    fi
}
trap on_error ERR

# run a command with its output in LOG_DIR/<name>.log
logged() {
    local name="$1"; shift
    CURRENT_LOG="$LOG_DIR/$name.log"
    "$@" > "$CURRENT_LOG" 2>&1
    CURRENT_LOG=""
}

default_jobs() {
    local kb gb cores jobs
    kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    gb=$(( kb / 1024 / 1024 ))
    cores=$(nproc)
    if (( gb < 10 )); then jobs=2                       # Jetson Orin Nano (7.3 GB + swap): the big files need ~2 GB each
    else jobs=$(( gb / 2 )); (( jobs > cores )) && jobs=$cores; fi
    (( jobs < 1 )) && jobs=1
    echo "$jobs"
}
JOBS="${JOBS:-$(default_jobs)}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer (got '$JOBS')"

# ---------------------------------------------------------------------------------------------------------------------
say "umiproc native build: fork $FORK_DIR, $JOBS jobs, OpenCV mode $UMIPROC_OPENCV ($(uname -m))"
[[ "$UMIPROC_OPENCV" =~ ^(auto|system|build)$ ]] || die "UMIPROC_OPENCV must be auto, system or build"
[[ -z "$PATCHES_UPTO" || "$PATCHES_UPTO" =~ ^[0-9]{4}$ ]] || die "PATCHES_UPTO must be a 4-digit patch number (got '$PATCHES_UPTO')"

# 1. tools and libraries -----------------------------------------------------------------------------------------------
missing=()
for tool in git cmake make g++ tar awk; do command -v "$tool" > /dev/null || missing+=("$tool"); done
((${#missing[@]} == 0)) || die "missing tools: ${missing[*]}
  Ubuntu / JetPack: sudo apt install build-essential cmake git"

if [[ -n "${OpenCV_DIR:-}" && ! -f "$OpenCV_DIR/OpenCVConfig.cmake" && ! -f "$OpenCV_DIR/opencv-config.cmake" ]]; then
    die "OpenCV_DIR=$OpenCV_DIR has no OpenCVConfig.cmake (CMake would silently use another OpenCV)"
fi

PROBE_DIR="$(mktemp -d)"
trap 'rm -rf "$PROBE_DIR"' EXIT
mkdir -p "$PROBE_DIR/src"
cat > "$PROBE_DIR/src/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.10)
project(umiproc_probe CXX)
set(missing "")
find_package(Eigen3 3.1.0 QUIET)
if(NOT Eigen3_FOUND AND NOT EIGEN3_FOUND)
  list(APPEND missing "Eigen3 >= 3.1 (libeigen3-dev)")
endif()
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL QUIET)
if(NOT OPENGL_FOUND)
  list(APPEND missing "OpenGL headers (libgl1-mesa-dev, libopengl-dev)")
endif()
find_package(GLEW QUIET)
if(NOT GLEW_FOUND)
  list(APPEND missing "GLEW (libglew-dev)")
endif()
find_package(Boost QUIET COMPONENTS serialization)
if(NOT Boost_FOUND)
  list(APPEND missing "Boost.Serialization (libboost-serialization-dev)")
endif()
find_package(OpenSSL QUIET)
if(NOT OPENSSL_FOUND)
  list(APPEND missing "OpenSSL libcrypto (libssl-dev)")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/missing.txt" "${missing}")
# OpenCV: version and whether the contrib aruco module (opencv2/aruco.hpp) is there
find_package(OpenCV QUIET)
set(aruco "no")
if(OpenCV_FOUND)
  foreach(d ${OpenCV_INCLUDE_DIRS})
    if(EXISTS "${d}/opencv2/aruco.hpp")
      set(aruco "yes")
    endif()
  endforeach()
endif()
file(WRITE "${CMAKE_BINARY_DIR}/opencv.txt" "found=${OpenCV_FOUND}\nversion=${OpenCV_VERSION}\ndir=${OpenCV_DIR}\naruco=${aruco}\n")
EOF
probe_args=()
[[ -n "${OpenCV_DIR:-}" ]] && probe_args+=("-DOpenCV_DIR=$OpenCV_DIR")
cmake -S "$PROBE_DIR/src" -B "$PROBE_DIR/build" "${probe_args[@]}" > "$PROBE_DIR/probe.log" 2>&1 \
    || { cat "$PROBE_DIR/probe.log" >&2; die "the dependency probe failed (cmake)"; }
probe_missing="$(cat "$PROBE_DIR/build/missing.txt")"
[[ -z "$probe_missing" ]] || die "missing libraries: ${probe_missing//;/, }
  Ubuntu / JetPack: sudo apt install libeigen3-dev libgl1-mesa-dev libopengl-dev libglew-dev \\
                    libboost-serialization-dev libssl-dev"
cv_found=$(sed -n 's/^found=//p' "$PROBE_DIR/build/opencv.txt")
cv_version=$(sed -n 's/^version=//p' "$PROBE_DIR/build/opencv.txt")
cv_dir=$(sed -n 's/^dir=//p' "$PROBE_DIR/build/opencv.txt")
cv_aruco=$(sed -n 's/^aruco=//p' "$PROBE_DIR/build/opencv.txt")

# 2. which OpenCV -----------------------------------------------------------------------------------------------------
version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]; }   # $1 >= $2
cv_ok=no
cv_why=""
if [[ ! "$cv_found" =~ ^(1|TRUE|ON|YES|Y)$ ]]; then cv_why="no system OpenCV found"
elif ! version_ge "$cv_version" "4.0"; then cv_why="system OpenCV $cv_version is older than 4.0"
elif version_ge "$cv_version" "4.7"; then cv_why="system OpenCV $cv_version has the 4.7+ aruco API the fork does not compile against"
elif [[ "$cv_aruco" != "yes" ]]; then cv_why="system OpenCV $cv_version has no contrib aruco module (opencv2/aruco.hpp)"
else cv_ok=yes; fi

OPENCV_ARGS=()
OPENCV_PREFIX="$FORK_DIR/Thirdparty/opencv-$OPENCV_VERSION/install"
case "$UMIPROC_OPENCV" in
    system)
        [[ "$cv_ok" == yes ]] || die "$cv_why
  Laptop (Ubuntu 22.04): sudo apt install libopencv-dev libopencv-contrib-dev
  Otherwise run with UMIPROC_OPENCV=build (or auto) to build OpenCV $OPENCV_VERSION + contrib for the fork"
        use_private_opencv=no ;;
    build) use_private_opencv=yes; cv_why="UMIPROC_OPENCV=build" ;;
    auto)  if [[ "$cv_ok" == yes ]]; then use_private_opencv=no; else use_private_opencv=yes; fi ;;
esac
if [[ "$use_private_opencv" == no ]]; then
    say "OpenCV: system $cv_version ($cv_dir), contrib aruco present"
    OPENCV_ARGS+=("-DOpenCV_DIR=$cv_dir")      # explicit, so a cached private OpenCV is not kept by mistake
else
    say "OpenCV: private $OPENCV_VERSION + contrib under $OPENCV_PREFIX ($cv_why)"
    OPENCV_ARGS+=("-DOpenCV_DIR=$OPENCV_PREFIX/lib/cmake/opencv4")
fi

# 3. the fork ---------------------------------------------------------------------------------------------------------
if [[ ! -e "$FORK_DIR" ]]; then
    say "cloning the fork into $FORK_DIR"
    mkdir -p "$(dirname "$FORK_DIR")"
    git clone --quiet "$FORK_URL" "$FORK_DIR"
    git -C "$FORK_DIR" checkout --quiet --detach "$FORK_COMMIT"
else
    git -C "$FORK_DIR" rev-parse --git-dir > /dev/null 2>&1 \
        || die "$FORK_DIR exists but is not a git checkout (move it away or set ORB_SLAM3_UMI_DIR)"
    head="$(git -C "$FORK_DIR" rev-parse HEAD)"
    [[ "$head" == "$FORK_COMMIT" ]] \
        || die "$FORK_DIR is at $head, not $FORK_COMMIT (check it out there, or set ORB_SLAM3_UMI_DIR)"
    say "fork present at ${FORK_COMMIT:0:7}"
fi
mkdir -p "$LOG_DIR"

pangolin_head="$(git -C "$FORK_DIR/$PANGOLIN_PATH" rev-parse HEAD 2>/dev/null || true)"
if [[ "$pangolin_head" != "$PANGOLIN_COMMIT" ]]; then
    if [[ -e "$FORK_DIR/$PANGOLIN_PATH/.git" ]]; then
        die "$FORK_DIR/$PANGOLIN_PATH is a checkout at ${pangolin_head:-?}, not $PANGOLIN_COMMIT"
    fi
    say "fetching Pangolin ${PANGOLIN_COMMIT:0:7} (https)"
    git -C "$FORK_DIR" config "submodule.$PANGOLIN_PATH.url" "$PANGOLIN_URL"
    logged pangolin_fetch git -C "$FORK_DIR" submodule update --init "$PANGOLIN_PATH"
    [[ "$(git -C "$FORK_DIR/$PANGOLIN_PATH" rev-parse HEAD)" == "$PANGOLIN_COMMIT" ]] \
        || die "Pangolin is not at $PANGOLIN_COMMIT after the submodule update"
else
    say "Pangolin present at ${PANGOLIN_COMMIT:0:7}"
fi

# 4. patches (each one either applies, or is already applied; anything else stops the build) --------------------------
shopt -s nullglob
patches=("$PATCH_DIR"/[0-9][0-9][0-9][0-9]-*.patch)
shopt -u nullglob
((${#patches[@]} > 0)) || die "no patches in $PATCH_DIR"

applied_patches=()      # the patches of the last apply_patches call
apply_patches() {       # fork_dir upto: apply every patch numbered <= upto ("" = all); patches above upto must not
    local dir="$1" upto="$2" p name f     # be applied, and nothing outside the patches may be changed
    local -A touched=()
    applied_patches=()
    for p in "${patches[@]}"; do
        name="$(basename "$p")"
        if [[ -n "$upto" && "${name:0:4}" > "$upto" ]]; then
            if git -C "$dir" apply --reverse --check "$p" 2> /dev/null; then
                die "$dir has patch $name applied, beyond PATCHES_UPTO=$upto (remove that folder and run again)"
            fi
            continue
        fi
        if git -C "$dir" apply --check "$p" 2> /dev/null; then
            git -C "$dir" apply "$p"
            say "patch $name: applied"
        elif git -C "$dir" apply --reverse --check "$p" 2> /dev/null; then
            say "patch $name: already applied"
        else
            die "patch $name neither applies nor is already applied: the fork's sources have other changes
  see: git -C $dir diff"
        fi
        applied_patches+=("$p")
        while IFS= read -r f; do touched["$f"]=1; done < <(sed -n 's|^+++ b/||p' "$p")
    done
    ((${#applied_patches[@]} > 0)) || die "no patch is numbered <= $upto"
    while IFS= read -r f; do
        [[ -n "${touched[$f]:-}" ]] \
            || die "$dir has local changes outside the patches ($f); restore it: git -C $dir checkout -- $f"
    done < <(git -C "$dir" diff --name-only)
}
apply_patches "$FORK_DIR" ""          # the main tree always carries every patch

# 5. private OpenCV (only when needed) ---------------------------------------------------------------------------------
if [[ "$use_private_opencv" == yes ]]; then
    cvroot="$FORK_DIR/Thirdparty/opencv-$OPENCV_VERSION"
    cv_args=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OPENCV_PREFIX"
        -DOPENCV_EXTRA_MODULES_PATH="$cvroot/opencv_contrib/modules" -DBUILD_LIST="$OPENCV_MODULES"
        -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF
        -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DWITH_CUDA=OFF
        -DWITH_IPP=OFF -DWITH_ADE=OFF -DWITH_OPENCL=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF
        -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_1394=OFF -DWITH_ITT=OFF -DWITH_PROTOBUF=OFF
        -DOPENCV_GENERATE_PKGCONFIG=OFF)
    stamp="$OPENCV_PREFIX/.umiproc_opencv"
    want="opencv=$OPENCV_COMMIT contrib=$OPENCV_CONTRIB_COMMIT $(printf '%s ' "${cv_args[@]}")"
    if [[ -f "$OPENCV_PREFIX/lib/cmake/opencv4/OpenCVConfig.cmake" && "$(cat "$stamp" 2>/dev/null)" == "$want" ]]; then
        say "private OpenCV $OPENCV_VERSION present"
    else
        mkdir -p "$cvroot"
        for spec in "opencv:$OPENCV_COMMIT" "opencv_contrib:$OPENCV_CONTRIB_COMMIT"; do
            repo="${spec%%:*}"; commit="${spec#*:}"
            if [[ "$(git -C "$cvroot/$repo" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
                say "fetching $repo $OPENCV_VERSION"
                rm -rf "$cvroot/$repo"
                logged "${repo}_fetch" git clone --quiet --depth 1 --branch "$OPENCV_VERSION" \
                    "https://github.com/opencv/$repo.git" "$cvroot/$repo"
                [[ "$(git -C "$cvroot/$repo" rev-parse HEAD)" == "$commit" ]] \
                    || die "$repo tag $OPENCV_VERSION is not at the pinned commit $commit"
            fi
        done
        say "building OpenCV $OPENCV_VERSION ($OPENCV_MODULES; takes a while with $JOBS jobs)"
        logged opencv_cmake cmake -S "$cvroot/opencv" -B "$cvroot/build" "${cv_args[@]}"
        logged opencv_build cmake --build "$cvroot/build" -j "$JOBS"
        logged opencv_install cmake --install "$cvroot/build"
        printf '%s' "$want" > "$stamp"
    fi
fi

# 6. third-party libraries, vocabulary, the fork's library, the front-end ----------------------------------------------
cmake_build() {   # name source build_dir [cmake args...]; then builds every target, or $TARGET when set
    local name="$1" src="$2" bld="$3"; shift 3
    say "building $name"
    logged "${name}_cmake" cmake -S "$src" -B "$bld" -DCMAKE_BUILD_TYPE=Release "$@"
    if [[ -n "${TARGET:-}" ]]; then
        logged "${name}_build" cmake --build "$bld" -j "$JOBS" --target "$TARGET"
    else
        logged "${name}_build" cmake --build "$bld" -j "$JOBS"
    fi
}
cmake_build DBoW2 "$FORK_DIR/Thirdparty/DBoW2" "$FORK_DIR/Thirdparty/DBoW2/build" "${OPENCV_ARGS[@]}"
cmake_build g2o "$FORK_DIR/Thirdparty/g2o" "$FORK_DIR/Thirdparty/g2o/build"
cmake_build Sophus "$FORK_DIR/Thirdparty/Sophus" "$FORK_DIR/Thirdparty/Sophus/build" \
    -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake_build Pangolin "$FORK_DIR/$PANGOLIN_PATH" "$FORK_DIR/$PANGOLIN_PATH/build" \
    -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_FFMPEG=OFF

VOC="$FORK_DIR/Vocabulary/ORBvoc.txt"
if [[ ! -s "$VOC" ]]; then          # unpacked to a temporary name first, so an interrupted run leaves no partial file
    say "unpacking the vocabulary"
    tar -xzOf "$FORK_DIR/Vocabulary/ORBvoc.txt.tar.gz" ORBvoc.txt > "$VOC.part"
    mv "$VOC.part" "$VOC"
else
    say "vocabulary present"
fi

PANGOLIN_ARGS=("-DPangolin_DIR=$FORK_DIR/$PANGOLIN_PATH/build")     # never another installed Pangolin
if [[ -z "$PATCHES_UPTO" ]]; then
    LIB_SRC="$FORK_DIR"                   # the fork's library builds into <source>/lib
    variant_patches=("${patches[@]}")
else
    # the variant: its own worktree of the fork at the same commit, with the patches up to PATCHES_UPTO, so its
    # libORB_SLAM3.so (in fork/lib) is a separate file the main build never writes
    LIB_SRC="$OUT_DIR/fork"
    if [[ ! -e "$LIB_SRC" ]]; then
        say "variant ${PATCHES_UPTO}: new worktree of the fork in $LIB_SRC"
        logged variant_worktree git -C "$FORK_DIR" worktree add --detach "$LIB_SRC" "$FORK_COMMIT"
    fi
    [[ "$(git -C "$LIB_SRC" rev-parse HEAD 2>/dev/null || true)" == "$FORK_COMMIT" ]] \
        || die "$LIB_SRC is not a checkout of the fork at $FORK_COMMIT (remove it and run again)"
    apply_patches "$LIB_SRC" "$PATCHES_UPTO"
    variant_patches=("${applied_patches[@]}")
    for lib in Thirdparty/DBoW2/lib/libDBoW2.so Thirdparty/g2o/lib/libg2o.so; do     # unpatched: shared
        mkdir -p "$LIB_SRC/$(dirname "$lib")"
        ln -sfn "$FORK_DIR/$lib" "$LIB_SRC/$lib"
    done
fi
TARGET=ORB_SLAM3 cmake_build ORB_SLAM3 "$LIB_SRC" "$LIB_SRC/build" "${PANGOLIN_ARGS[@]}" "${OPENCV_ARGS[@]}"
cmake_build umiproc_slam "$FRONTEND_SRC" "$OUT_DIR" -DORB_SLAM3_DIR="$LIB_SRC" "${PANGOLIN_ARGS[@]}" \
    "${OPENCV_ARGS[@]}"

# 7. check and record ----------------------------------------------------------------------------------------------------
BIN="$OUT_DIR/umiproc_slam"
[[ -x "$BIN" ]] || die "no binary at $BIN"
ldd "$BIN" > "$LOG_DIR/ldd.log"
if grep -q "not found" "$LOG_DIR/ldd.log"; then
    grep "not found" "$LOG_DIR/ldd.log" >&2
    die "$BIN has unresolved libraries"
fi
lib_used="$(awk '/libORB_SLAM3\.so/ {print $3}' "$LOG_DIR/ldd.log")"
[[ -n "$lib_used" && "$(readlink -f "$lib_used")" == "$(readlink -f "$LIB_SRC/lib/libORB_SLAM3.so")" ]] \
    || die "$BIN loads libORB_SLAM3.so from '$lib_used', not $LIB_SRC/lib (check LD_LIBRARY_PATH)"
"$BIN" --help > /dev/null || die "$BIN --help failed"
opencv_used="$(ldd "$LIB_SRC/lib/libORB_SLAM3.so" | awk '/libopencv_core/ {print $3}')"
{
    echo "fork $FORK_URL $FORK_COMMIT"
    echo "pangolin $PANGOLIN_COMMIT"
    if [[ -n "$PATCHES_UPTO" ]]; then echo "variant patches up to $PATCHES_UPTO (sources $LIB_SRC)"; fi
    echo "library $LIB_SRC/lib/libORB_SLAM3.so"
    for p in "${variant_patches[@]}"; do echo "patch $(sha256sum "$p" | cut -c1-16) $(basename "$p")"; done
    echo "opencv $opencv_used"
    echo "machine $(uname -m) $(uname -r)"
    echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/BUILD_INFO"
say "done"
echo "  binary:     $BIN"
echo "  vocabulary: $VOC"
echo "  build info: $OUT_DIR/BUILD_INFO"

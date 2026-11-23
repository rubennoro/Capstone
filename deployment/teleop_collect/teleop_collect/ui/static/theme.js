// Teleop Collect: the theme, set before the first paint (dark unless this browser chose light in the header's
// toggle). Loaded in <head>; app.js owns the toggle. Plain JS, offline.
"use strict";
(function () {
  let theme = "dark";
  try {
    if (localStorage.getItem("teleop_collect.theme") === "light") theme = "light";
  } catch (e) { /* storage blocked: dark */ }
  document.documentElement.dataset.theme = theme;
})();

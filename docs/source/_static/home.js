document.addEventListener("DOMContentLoaded", function () {
  const pill = document.querySelector(".mp-home-pip-snippet .highlight");
  if (!pill) return;

  pill.addEventListener("click", function (e) {
    const btn = pill.querySelector("button.copybtn");
    if (btn && !btn.contains(e.target)) {
      btn.click();
    }
  });
});

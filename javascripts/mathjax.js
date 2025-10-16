/* MathJax Configuration for AG News Text Classification Documentation */
/* Author: Võ Hải Dũng */

window.MathJax = {
  tex: {
    inlineMath: [["\KATEX_INLINE_OPEN", "\KATEX_INLINE_CLOSE"], ["$", "$"]],
    displayMath: [["\```math", "\```"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
});

/** A phone or tablet: focusing a field on load raises the keyboard and scrolls the page away. */
export function isTouchDevice(): boolean {
  try {
    return window.matchMedia("(pointer: coarse)").matches;
  } catch {
    return false;
  }
}

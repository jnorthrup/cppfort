void t: type = {
  operator[](this, _) { } }

main: () -> int = {
  (x := t()) { x[:() -> _ = 0]; }
  (x := t()) { x[:() -> _ = 0;]; }

  assert(!(:() = 0; is int));

  return :i32 = 0;
}auto x = = :i32 = 0;
int32_t y = 0;

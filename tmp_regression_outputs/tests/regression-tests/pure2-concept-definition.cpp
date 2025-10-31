<T> concept arithmetic = std::integral<T> || std::floating_point<T>;
void main() { assert<testing>( arithmetic<i32> );
  assert<testing>( arithmetic<float> ); }

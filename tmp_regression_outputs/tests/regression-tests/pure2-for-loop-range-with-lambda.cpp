#include <iostream>
#include <array>
auto&& first(forward f, forward _...) { return f; }

int main(args) { const std::array ints = (1, 2, 3, 4, 5);
   // OK
   for ints.first() do (i) {
      std::cout << i;
   }

   // OK
   for ints.first(1) do (i)  {
      std::cout << i;
   }

   // Used to cause Error
   for ints.first(:(forward x) = x) do (i)   {
      std::cout << i;
   }

   auto // OK
   temp = ints.first(:(forward x) = x);
   for temp do (i) {
      std::cout << i;
   }

   ( ) for = args$;()  do (i)  _ = i;
   (forward x) for = x;(args)  do (j)  _ = j;
   (forward x) for = x;( ) (args)  next _ = = args$;()  do (k)  _ = k;
   (forward x) for = x;(forward x) (args)  next _ = = x;(args)  do (l)  _ = l; }


/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar066-min-function/ar066-min-function_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_minii>:
100000360: 6b01001f    	cmp	w0, w1
100000364: 1a81b000    	csel	w0, w0, w1, lt
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800060    	mov	w0, #0x3                ; =3
100000370: d65f03c0    	ret

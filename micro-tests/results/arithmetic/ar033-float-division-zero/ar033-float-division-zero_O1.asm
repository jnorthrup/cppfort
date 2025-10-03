
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar033-float-division-zero/ar033-float-division-zero_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_float_div_zerof>:
100000360: 2f00e401    	movi	d1, #0000000000000000
100000364: 1e211800    	fdiv	s0, s0, s1
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret

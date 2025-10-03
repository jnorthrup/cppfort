
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar030-float-unary-minus/ar030-float-unary-minus_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_float_negf>:
100000360: 1e214000    	fneg	s0, s0
100000364: d65f03c0    	ret

0000000100000368 <_main>:
100000368: 12800080    	mov	w0, #-0x5               ; =-5
10000036c: d65f03c0    	ret

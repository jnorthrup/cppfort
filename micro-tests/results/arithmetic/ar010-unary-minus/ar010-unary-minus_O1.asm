
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar010-unary-minus/ar010-unary-minus_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_unary_minusi>:
100000360: 4b0003e0    	neg	w0, w0
100000364: d65f03c0    	ret

0000000100000368 <_main>:
100000368: 12800080    	mov	w0, #-0x5               ; =-5
10000036c: d65f03c0    	ret

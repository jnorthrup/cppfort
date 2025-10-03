
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar067-max-function/ar067-max-function_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_maxii>:
100000360: 6b01001f    	cmp	w0, w1
100000364: 1a81c000    	csel	w0, w0, w1, gt
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 528000a0    	mov	w0, #0x5                ; =5
100000370: d65f03c0    	ret

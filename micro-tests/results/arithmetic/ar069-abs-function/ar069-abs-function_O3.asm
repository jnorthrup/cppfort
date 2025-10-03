
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar069-abs-function/ar069-abs-function_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_absi>:
100000360: 7100001f    	cmp	w0, #0x0
100000364: 5a805400    	cneg	w0, w0, mi
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 528000a0    	mov	w0, #0x5                ; =5
100000370: d65f03c0    	ret

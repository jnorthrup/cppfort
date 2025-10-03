
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar034-float-nan/ar034-float-nan_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_nanv>:
100000360: 52aff808    	mov	w8, #0x7fc00000         ; =2143289344
100000364: 1e270100    	fmov	s0, w8
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret

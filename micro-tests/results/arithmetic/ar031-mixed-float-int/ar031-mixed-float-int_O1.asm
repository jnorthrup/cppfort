
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar031-mixed-float-int/ar031-mixed-float-int_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_mixed_typesif>:
100000360: 1e220001    	scvtf	s1, w0
100000364: 1e202820    	fadd	s0, s1, s0
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800100    	mov	w0, #0x8                ; =8
100000370: d65f03c0    	ret

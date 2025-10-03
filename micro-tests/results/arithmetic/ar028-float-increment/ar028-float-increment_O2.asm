
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar028-float-increment/ar028-float-increment_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_float_incf>:
100000360: 1e2e1001    	fmov	s1, #1.00000000
100000364: 1e212800    	fadd	s0, s0, s1
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 528000c0    	mov	w0, #0x6                ; =6
100000370: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar028-float-increment/ar028-float-increment_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_float_incf>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: bd000fe0    	str	s0, [sp, #0xc]
100000368: bd400fe0    	ldr	s0, [sp, #0xc]
10000036c: 1e2e1001    	fmov	s1, #1.00000000
100000370: 1e212800    	fadd	s0, s0, s1
100000374: bd000fe0    	str	s0, [sp, #0xc]
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 1e22d000    	fmov	s0, #5.50000000
100000394: 97fffff3    	bl	0x100000360 <__Z14test_float_incf>
100000398: 1e380000    	fcvtzs	w0, s0
10000039c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a0: 910083ff    	add	sp, sp, #0x20
1000003a4: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar021-add-float/ar021-add-float_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_add_floatff>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: bd000fe0    	str	s0, [sp, #0xc]
100000368: bd000be1    	str	s1, [sp, #0x8]
10000036c: bd400fe0    	ldr	s0, [sp, #0xc]
100000370: bd400be1    	ldr	s1, [sp, #0x8]
100000374: 1e212800    	fadd	s0, s0, s1
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 1e219000    	fmov	s0, #3.50000000
100000394: 1e209001    	fmov	s1, #2.50000000
100000398: 97fffff2    	bl	0x100000360 <__Z14test_add_floatff>
10000039c: 1e380000    	fcvtzs	w0, s0
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret

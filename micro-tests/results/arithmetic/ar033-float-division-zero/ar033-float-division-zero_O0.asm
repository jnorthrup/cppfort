
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar033-float-division-zero/ar033-float-division-zero_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_float_div_zerof>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: bd000fe0    	str	s0, [sp, #0xc]
100000368: bd400fe0    	ldr	s0, [sp, #0xc]
10000036c: 2f00e401    	movi	d1, #0000000000000000
100000370: 1e211800    	fdiv	s0, s0, s1
100000374: 910043ff    	add	sp, sp, #0x10
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: d10083ff    	sub	sp, sp, #0x20
100000380: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000384: 910043fd    	add	x29, sp, #0x10
100000388: 52800008    	mov	w8, #0x0                ; =0
10000038c: b90007e8    	str	w8, [sp, #0x4]
100000390: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000394: 1e2e1000    	fmov	s0, #1.00000000
100000398: 97fffff2    	bl	0x100000360 <__Z19test_float_div_zerof>
10000039c: b94007e8    	ldr	w8, [sp, #0x4]
1000003a0: bd000be0    	str	s0, [sp, #0x8]
1000003a4: bd400be0    	ldr	s0, [sp, #0x8]
1000003a8: 52a88f49    	mov	w9, #0x447a0000         ; =1148846080
1000003ac: 1e270121    	fmov	s1, w9
1000003b0: 1e212000    	fcmp	s0, s1
1000003b4: 1a9fd500    	csinc	w0, w8, wzr, le
1000003b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003bc: 910083ff    	add	sp, sp, #0x20
1000003c0: d65f03c0    	ret

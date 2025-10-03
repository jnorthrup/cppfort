
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar034-float-nan/ar034-float-nan_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_nanv>:
100000360: 52aff808    	mov	w8, #0x7fc00000         ; =2143289344
100000364: 1e270100    	fmov	s0, w8
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: d10083ff    	sub	sp, sp, #0x20
100000370: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000374: 910043fd    	add	x29, sp, #0x10
100000378: 52800008    	mov	w8, #0x0                ; =0
10000037c: b90007e8    	str	w8, [sp, #0x4]
100000380: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000384: 97fffff7    	bl	0x100000360 <__Z8test_nanv>
100000388: b94007e8    	ldr	w8, [sp, #0x4]
10000038c: bd000be0    	str	s0, [sp, #0x8]
100000390: bd400be0    	ldr	s0, [sp, #0x8]
100000394: bd400be1    	ldr	s1, [sp, #0x8]
100000398: 1e212000    	fcmp	s0, s1
10000039c: 1a9f0500    	csinc	w0, w8, wzr, eq
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret

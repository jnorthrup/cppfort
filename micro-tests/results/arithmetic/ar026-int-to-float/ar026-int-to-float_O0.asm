
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar026-int-to-float/ar026-int-to-float_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_int_to_floati>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: bd400fe0    	ldr	s0, [sp, #0xc]
10000036c: 5e21d800    	scvtf	s0, s0
100000370: 910043ff    	add	sp, sp, #0x10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: d10083ff    	sub	sp, sp, #0x20
10000037c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000380: 910043fd    	add	x29, sp, #0x10
100000384: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000388: 528000a0    	mov	w0, #0x5                ; =5
10000038c: 97fffff5    	bl	0x100000360 <__Z17test_int_to_floati>
100000390: 1e380000    	fcvtzs	w0, s0
100000394: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000398: 910083ff    	add	sp, sp, #0x20
10000039c: d65f03c0    	ret

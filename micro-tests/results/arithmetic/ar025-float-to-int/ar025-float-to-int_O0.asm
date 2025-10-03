
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar025-float-to-int/ar025-float-to-int_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_float_to_intf>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: bd000fe0    	str	s0, [sp, #0xc]
100000368: bd400fe0    	ldr	s0, [sp, #0xc]
10000036c: 1e380000    	fcvtzs	w0, s0
100000370: 910043ff    	add	sp, sp, #0x10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: d10083ff    	sub	sp, sp, #0x20
10000037c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000380: 910043fd    	add	x29, sp, #0x10
100000384: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000388: 529999a8    	mov	w8, #0xcccd             ; =52429
10000038c: 72a80d88    	movk	w8, #0x406c, lsl #16
100000390: 1e270100    	fmov	s0, w8
100000394: 97fffff3    	bl	0x100000360 <__Z17test_float_to_intf>
100000398: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000039c: 910083ff    	add	sp, sp, #0x20
1000003a0: d65f03c0    	ret

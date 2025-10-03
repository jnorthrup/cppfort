
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar029-double-compound/ar029-double-compound_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_double_compounddd>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: fd0007e0    	str	d0, [sp, #0x8]
100000368: fd0003e1    	str	d1, [sp]
10000036c: fd4003e1    	ldr	d1, [sp]
100000370: fd4007e0    	ldr	d0, [sp, #0x8]
100000374: 1e612800    	fadd	d0, d0, d1
100000378: fd0007e0    	str	d0, [sp, #0x8]
10000037c: fd4007e0    	ldr	d0, [sp, #0x8]
100000380: 1e601001    	fmov	d1, #2.00000000
100000384: 1e610800    	fmul	d0, d0, d1
100000388: fd0007e0    	str	d0, [sp, #0x8]
10000038c: fd4007e0    	ldr	d0, [sp, #0x8]
100000390: 910043ff    	add	sp, sp, #0x10
100000394: d65f03c0    	ret

0000000100000398 <_main>:
100000398: d10083ff    	sub	sp, sp, #0x20
10000039c: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a0: 910043fd    	add	x29, sp, #0x10
1000003a4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a8: 1e611000    	fmov	d0, #3.00000000
1000003ac: 1e601001    	fmov	d1, #2.00000000
1000003b0: 97ffffec    	bl	0x100000360 <__Z20test_double_compounddd>
1000003b4: 1e780000    	fcvtzs	w0, d0
1000003b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003bc: 910083ff    	add	sp, sp, #0x20
1000003c0: d65f03c0    	ret

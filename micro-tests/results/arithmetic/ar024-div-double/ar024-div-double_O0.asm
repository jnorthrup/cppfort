
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar024-div-double/ar024-div-double_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_div_doubledd>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: fd000be0    	str	d0, [sp, #0x10]
100000368: fd0007e1    	str	d1, [sp, #0x8]
10000036c: fd4007e0    	ldr	d0, [sp, #0x8]
100000370: 1e602008    	fcmp	d0, #0.0
100000374: 540000a1    	b.ne	0x100000388 <__Z15test_div_doubledd+0x28>
100000378: 14000001    	b	0x10000037c <__Z15test_div_doubledd+0x1c>
10000037c: 2f00e400    	movi	d0, #0000000000000000
100000380: fd000fe0    	str	d0, [sp, #0x18]
100000384: 14000006    	b	0x10000039c <__Z15test_div_doubledd+0x3c>
100000388: fd400be0    	ldr	d0, [sp, #0x10]
10000038c: fd4007e1    	ldr	d1, [sp, #0x8]
100000390: 1e611800    	fdiv	d0, d0, d1
100000394: fd000fe0    	str	d0, [sp, #0x18]
100000398: 14000001    	b	0x10000039c <__Z15test_div_doubledd+0x3c>
10000039c: fd400fe0    	ldr	d0, [sp, #0x18]
1000003a0: 910083ff    	add	sp, sp, #0x20
1000003a4: d65f03c0    	ret

00000001000003a8 <_main>:
1000003a8: d10083ff    	sub	sp, sp, #0x20
1000003ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b0: 910043fd    	add	x29, sp, #0x10
1000003b4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b8: 1e669000    	fmov	d0, #20.00000000
1000003bc: 1e621001    	fmov	d1, #4.00000000
1000003c0: 97ffffe8    	bl	0x100000360 <__Z15test_div_doubledd>
1000003c4: 1e780000    	fcvtzs	w0, d0
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret

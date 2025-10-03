
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf015-ternary-complex-expr/cf015-ternary-complex-expr_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_ternary_complexii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: 6b090108    	subs	w8, w8, w9
100000378: 540000ed    	b.le	0x100000394 <__Z20test_ternary_complexii+0x34>
10000037c: 14000001    	b	0x100000380 <__Z20test_ternary_complexii+0x20>
100000380: b9400fe8    	ldr	w8, [sp, #0xc]
100000384: b9400be9    	ldr	w9, [sp, #0x8]
100000388: 0b090108    	add	w8, w8, w9
10000038c: b90007e8    	str	w8, [sp, #0x4]
100000390: 14000006    	b	0x1000003a8 <__Z20test_ternary_complexii+0x48>
100000394: b9400fe8    	ldr	w8, [sp, #0xc]
100000398: b9400be9    	ldr	w9, [sp, #0x8]
10000039c: 6b090108    	subs	w8, w8, w9
1000003a0: b90007e8    	str	w8, [sp, #0x4]
1000003a4: 14000001    	b	0x1000003a8 <__Z20test_ternary_complexii+0x48>
1000003a8: b94007e0    	ldr	w0, [sp, #0x4]
1000003ac: 910043ff    	add	sp, sp, #0x10
1000003b0: d65f03c0    	ret

00000001000003b4 <_main>:
1000003b4: d10083ff    	sub	sp, sp, #0x20
1000003b8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003bc: 910043fd    	add	x29, sp, #0x10
1000003c0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c4: 52800140    	mov	w0, #0xa                ; =10
1000003c8: 52800061    	mov	w1, #0x3                ; =3
1000003cc: 97ffffe5    	bl	0x100000360 <__Z20test_ternary_complexii>
1000003d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d4: 910083ff    	add	sp, sp, #0x20
1000003d8: d65f03c0    	ret

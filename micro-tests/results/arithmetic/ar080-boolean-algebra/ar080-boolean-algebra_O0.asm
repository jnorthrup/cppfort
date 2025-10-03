
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar080-boolean-algebra/ar080-boolean-algebra_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_boolean_algebrabbb>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: 39007fe0    	strb	w0, [sp, #0x1f]
100000368: 39007be1    	strb	w1, [sp, #0x1e]
10000036c: 390077e2    	strb	w2, [sp, #0x1d]
100000370: 39407fe8    	ldrb	w8, [sp, #0x1f]
100000374: 52800009    	mov	w9, #0x0                ; =0
100000378: b9001be9    	str	w9, [sp, #0x18]
10000037c: 360001a8    	tbz	w8, #0x0, 0x1000003b0 <__Z20test_boolean_algebrabbb+0x50>
100000380: 14000001    	b	0x100000384 <__Z20test_boolean_algebrabbb+0x24>
100000384: 39407be8    	ldrb	w8, [sp, #0x1e]
100000388: 52800029    	mov	w9, #0x1                ; =1
10000038c: b90017e9    	str	w9, [sp, #0x14]
100000390: 370000a8    	tbnz	w8, #0x0, 0x1000003a4 <__Z20test_boolean_algebrabbb+0x44>
100000394: 14000001    	b	0x100000398 <__Z20test_boolean_algebrabbb+0x38>
100000398: 394077e8    	ldrb	w8, [sp, #0x1d]
10000039c: b90017e8    	str	w8, [sp, #0x14]
1000003a0: 14000001    	b	0x1000003a4 <__Z20test_boolean_algebrabbb+0x44>
1000003a4: b94017e8    	ldr	w8, [sp, #0x14]
1000003a8: b9001be8    	str	w8, [sp, #0x18]
1000003ac: 14000001    	b	0x1000003b0 <__Z20test_boolean_algebrabbb+0x50>
1000003b0: b9401be8    	ldr	w8, [sp, #0x18]
1000003b4: 12000108    	and	w8, w8, #0x1
1000003b8: b90013e8    	str	w8, [sp, #0x10]
1000003bc: 39407fe8    	ldrb	w8, [sp, #0x1f]
1000003c0: 360000e8    	tbz	w8, #0x0, 0x1000003dc <__Z20test_boolean_algebrabbb+0x7c>
1000003c4: 14000001    	b	0x1000003c8 <__Z20test_boolean_algebrabbb+0x68>
1000003c8: 39407be8    	ldrb	w8, [sp, #0x1e]
1000003cc: 52800029    	mov	w9, #0x1                ; =1
1000003d0: b9000fe9    	str	w9, [sp, #0xc]
1000003d4: 370001a8    	tbnz	w8, #0x0, 0x100000408 <__Z20test_boolean_algebrabbb+0xa8>
1000003d8: 14000001    	b	0x1000003dc <__Z20test_boolean_algebrabbb+0x7c>
1000003dc: 39407fe8    	ldrb	w8, [sp, #0x1f]
1000003e0: 52800009    	mov	w9, #0x0                ; =0
1000003e4: b9000be9    	str	w9, [sp, #0x8]
1000003e8: 360000a8    	tbz	w8, #0x0, 0x1000003fc <__Z20test_boolean_algebrabbb+0x9c>
1000003ec: 14000001    	b	0x1000003f0 <__Z20test_boolean_algebrabbb+0x90>
1000003f0: 394077e8    	ldrb	w8, [sp, #0x1d]
1000003f4: b9000be8    	str	w8, [sp, #0x8]
1000003f8: 14000001    	b	0x1000003fc <__Z20test_boolean_algebrabbb+0x9c>
1000003fc: b9400be8    	ldr	w8, [sp, #0x8]
100000400: b9000fe8    	str	w8, [sp, #0xc]
100000404: 14000001    	b	0x100000408 <__Z20test_boolean_algebrabbb+0xa8>
100000408: b94013e8    	ldr	w8, [sp, #0x10]
10000040c: b9400fe9    	ldr	w9, [sp, #0xc]
100000410: 12000129    	and	w9, w9, #0x1
100000414: 6b090108    	subs	w8, w8, w9
100000418: 1a9f17e0    	cset	w0, eq
10000041c: 910083ff    	add	sp, sp, #0x20
100000420: d65f03c0    	ret

0000000100000424 <_main>:
100000424: d10083ff    	sub	sp, sp, #0x20
100000428: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000042c: 910043fd    	add	x29, sp, #0x10
100000430: 52800009    	mov	w9, #0x0                ; =0
100000434: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000438: 52800028    	mov	w8, #0x1                ; =1
10000043c: 12000100    	and	w0, w8, #0x1
100000440: 12000121    	and	w1, w9, #0x1
100000444: 12000102    	and	w2, w8, #0x1
100000448: 97ffffc6    	bl	0x100000360 <__Z20test_boolean_algebrabbb>
10000044c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000450: 910083ff    	add	sp, sp, #0x20
100000454: d65f03c0    	ret

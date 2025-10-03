
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf071-break-nested-loops/cf071-break-nested-loops_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_break_nestedv>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: 12800008    	mov	w8, #-0x1               ; =-1
100000368: b9001fe8    	str	w8, [sp, #0x1c]
10000036c: 39006fff    	strb	wzr, [sp, #0x1b]
100000370: b90017ff    	str	wzr, [sp, #0x14]
100000374: 14000001    	b	0x100000378 <__Z17test_break_nestedv+0x18>
100000378: b94017e9    	ldr	w9, [sp, #0x14]
10000037c: 52800008    	mov	w8, #0x0                ; =0
100000380: 71002929    	subs	w9, w9, #0xa
100000384: b9000fe8    	str	w8, [sp, #0xc]
100000388: 540000ca    	b.ge	0x1000003a0 <__Z17test_break_nestedv+0x40>
10000038c: 14000001    	b	0x100000390 <__Z17test_break_nestedv+0x30>
100000390: 39406fe8    	ldrb	w8, [sp, #0x1b]
100000394: 52000108    	eor	w8, w8, #0x1
100000398: b9000fe8    	str	w8, [sp, #0xc]
10000039c: 14000001    	b	0x1000003a0 <__Z17test_break_nestedv+0x40>
1000003a0: b9400fe8    	ldr	w8, [sp, #0xc]
1000003a4: 36000428    	tbz	w8, #0x0, 0x100000428 <__Z17test_break_nestedv+0xc8>
1000003a8: 14000001    	b	0x1000003ac <__Z17test_break_nestedv+0x4c>
1000003ac: b90013ff    	str	wzr, [sp, #0x10]
1000003b0: 14000001    	b	0x1000003b4 <__Z17test_break_nestedv+0x54>
1000003b4: b94013e8    	ldr	w8, [sp, #0x10]
1000003b8: 71002908    	subs	w8, w8, #0xa
1000003bc: 540002ca    	b.ge	0x100000414 <__Z17test_break_nestedv+0xb4>
1000003c0: 14000001    	b	0x1000003c4 <__Z17test_break_nestedv+0x64>
1000003c4: b94017e8    	ldr	w8, [sp, #0x14]
1000003c8: b94013e9    	ldr	w9, [sp, #0x10]
1000003cc: 1b097d08    	mul	w8, w8, w9
1000003d0: 71005108    	subs	w8, w8, #0x14
1000003d4: 54000161    	b.ne	0x100000400 <__Z17test_break_nestedv+0xa0>
1000003d8: 14000001    	b	0x1000003dc <__Z17test_break_nestedv+0x7c>
1000003dc: b94017e8    	ldr	w8, [sp, #0x14]
1000003e0: 52800149    	mov	w9, #0xa                ; =10
1000003e4: 1b097d08    	mul	w8, w8, w9
1000003e8: b94013e9    	ldr	w9, [sp, #0x10]
1000003ec: 0b090108    	add	w8, w8, w9
1000003f0: b9001fe8    	str	w8, [sp, #0x1c]
1000003f4: 52800028    	mov	w8, #0x1                ; =1
1000003f8: 39006fe8    	strb	w8, [sp, #0x1b]
1000003fc: 14000006    	b	0x100000414 <__Z17test_break_nestedv+0xb4>
100000400: 14000001    	b	0x100000404 <__Z17test_break_nestedv+0xa4>
100000404: b94013e8    	ldr	w8, [sp, #0x10]
100000408: 11000508    	add	w8, w8, #0x1
10000040c: b90013e8    	str	w8, [sp, #0x10]
100000410: 17ffffe9    	b	0x1000003b4 <__Z17test_break_nestedv+0x54>
100000414: 14000001    	b	0x100000418 <__Z17test_break_nestedv+0xb8>
100000418: b94017e8    	ldr	w8, [sp, #0x14]
10000041c: 11000508    	add	w8, w8, #0x1
100000420: b90017e8    	str	w8, [sp, #0x14]
100000424: 17ffffd5    	b	0x100000378 <__Z17test_break_nestedv+0x18>
100000428: b9401fe0    	ldr	w0, [sp, #0x1c]
10000042c: 910083ff    	add	sp, sp, #0x20
100000430: d65f03c0    	ret

0000000100000434 <_main>:
100000434: d10083ff    	sub	sp, sp, #0x20
100000438: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000043c: 910043fd    	add	x29, sp, #0x10
100000440: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000444: 97ffffc7    	bl	0x100000360 <__Z17test_break_nestedv>
100000448: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000044c: 910083ff    	add	sp, sp, #0x20
100000450: d65f03c0    	ret

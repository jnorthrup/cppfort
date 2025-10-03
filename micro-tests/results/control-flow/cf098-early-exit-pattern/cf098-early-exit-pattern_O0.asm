
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf098-early-exit-pattern/cf098-early-exit-pattern_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_early_exitsiii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007e1    	str	w1, [sp, #0x4]
10000036c: b90003e2    	str	w2, [sp]
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 36f800a8    	tbz	w8, #0x1f, 0x100000388 <__Z16test_early_exitsiii+0x28>
100000378: 14000001    	b	0x10000037c <__Z16test_early_exitsiii+0x1c>
10000037c: 12800008    	mov	w8, #-0x1               ; =-1
100000380: b9000fe8    	str	w8, [sp, #0xc]
100000384: 1400001d    	b	0x1000003f8 <__Z16test_early_exitsiii+0x98>
100000388: b94007e8    	ldr	w8, [sp, #0x4]
10000038c: 36f800a8    	tbz	w8, #0x1f, 0x1000003a0 <__Z16test_early_exitsiii+0x40>
100000390: 14000001    	b	0x100000394 <__Z16test_early_exitsiii+0x34>
100000394: 12800028    	mov	w8, #-0x2               ; =-2
100000398: b9000fe8    	str	w8, [sp, #0xc]
10000039c: 14000017    	b	0x1000003f8 <__Z16test_early_exitsiii+0x98>
1000003a0: b94003e8    	ldr	w8, [sp]
1000003a4: 36f800a8    	tbz	w8, #0x1f, 0x1000003b8 <__Z16test_early_exitsiii+0x58>
1000003a8: 14000001    	b	0x1000003ac <__Z16test_early_exitsiii+0x4c>
1000003ac: 12800048    	mov	w8, #-0x3               ; =-3
1000003b0: b9000fe8    	str	w8, [sp, #0xc]
1000003b4: 14000011    	b	0x1000003f8 <__Z16test_early_exitsiii+0x98>
1000003b8: b9400be8    	ldr	w8, [sp, #0x8]
1000003bc: b94007e9    	ldr	w9, [sp, #0x4]
1000003c0: 0b090108    	add	w8, w8, w9
1000003c4: b94003e9    	ldr	w9, [sp]
1000003c8: 0b090108    	add	w8, w8, w9
1000003cc: 35000088    	cbnz	w8, 0x1000003dc <__Z16test_early_exitsiii+0x7c>
1000003d0: 14000001    	b	0x1000003d4 <__Z16test_early_exitsiii+0x74>
1000003d4: b9000fff    	str	wzr, [sp, #0xc]
1000003d8: 14000008    	b	0x1000003f8 <__Z16test_early_exitsiii+0x98>
1000003dc: b9400be8    	ldr	w8, [sp, #0x8]
1000003e0: b94007e9    	ldr	w9, [sp, #0x4]
1000003e4: 1b097d08    	mul	w8, w8, w9
1000003e8: b94003e9    	ldr	w9, [sp]
1000003ec: 1b097d08    	mul	w8, w8, w9
1000003f0: b9000fe8    	str	w8, [sp, #0xc]
1000003f4: 14000001    	b	0x1000003f8 <__Z16test_early_exitsiii+0x98>
1000003f8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003fc: 910043ff    	add	sp, sp, #0x10
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: d10083ff    	sub	sp, sp, #0x20
100000408: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000040c: 910043fd    	add	x29, sp, #0x10
100000410: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000414: 52800040    	mov	w0, #0x2                ; =2
100000418: 52800061    	mov	w1, #0x3                ; =3
10000041c: 52800082    	mov	w2, #0x4                ; =4
100000420: 97ffffd0    	bl	0x100000360 <__Z16test_early_exitsiii>
100000424: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000428: 910083ff    	add	sp, sp, #0x20
10000042c: d65f03c0    	ret

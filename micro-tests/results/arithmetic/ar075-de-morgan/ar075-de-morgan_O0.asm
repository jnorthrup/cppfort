
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar075-de-morgan/ar075-de-morgan_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_de_morganbb>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39003fe0    	strb	w0, [sp, #0xf]
100000368: 39003be1    	strb	w1, [sp, #0xe]
10000036c: 39403fe8    	ldrb	w8, [sp, #0xf]
100000370: 52800009    	mov	w9, #0x0                ; =0
100000374: b9000be9    	str	w9, [sp, #0x8]
100000378: 360000a8    	tbz	w8, #0x0, 0x10000038c <__Z14test_de_morganbb+0x2c>
10000037c: 14000001    	b	0x100000380 <__Z14test_de_morganbb+0x20>
100000380: 39403be8    	ldrb	w8, [sp, #0xe]
100000384: b9000be8    	str	w8, [sp, #0x8]
100000388: 14000001    	b	0x10000038c <__Z14test_de_morganbb+0x2c>
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: 52000108    	eor	w8, w8, #0x1
100000394: 12000108    	and	w8, w8, #0x1
100000398: b90003e8    	str	w8, [sp]
10000039c: 39403fe8    	ldrb	w8, [sp, #0xf]
1000003a0: 52800029    	mov	w9, #0x1                ; =1
1000003a4: b90007e9    	str	w9, [sp, #0x4]
1000003a8: 360000c8    	tbz	w8, #0x0, 0x1000003c0 <__Z14test_de_morganbb+0x60>
1000003ac: 14000001    	b	0x1000003b0 <__Z14test_de_morganbb+0x50>
1000003b0: 39403be8    	ldrb	w8, [sp, #0xe]
1000003b4: 52000108    	eor	w8, w8, #0x1
1000003b8: b90007e8    	str	w8, [sp, #0x4]
1000003bc: 14000001    	b	0x1000003c0 <__Z14test_de_morganbb+0x60>
1000003c0: b94003e8    	ldr	w8, [sp]
1000003c4: b94007e9    	ldr	w9, [sp, #0x4]
1000003c8: 12000129    	and	w9, w9, #0x1
1000003cc: 6b090108    	subs	w8, w8, w9
1000003d0: 1a9f17e0    	cset	w0, eq
1000003d4: 910043ff    	add	sp, sp, #0x10
1000003d8: d65f03c0    	ret

00000001000003dc <_main>:
1000003dc: d10083ff    	sub	sp, sp, #0x20
1000003e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e4: 910043fd    	add	x29, sp, #0x10
1000003e8: 52800008    	mov	w8, #0x0                ; =0
1000003ec: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003f0: 52800029    	mov	w9, #0x1                ; =1
1000003f4: 12000120    	and	w0, w9, #0x1
1000003f8: 12000101    	and	w1, w8, #0x1
1000003fc: 97ffffd9    	bl	0x100000360 <__Z14test_de_morganbb>
100000400: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000404: 910083ff    	add	sp, sp, #0x20
100000408: d65f03c0    	ret

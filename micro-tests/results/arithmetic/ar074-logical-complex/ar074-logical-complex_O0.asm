
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar074-logical-complex/ar074-logical-complex_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_logical_complexbbb>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39003fe0    	strb	w0, [sp, #0xf]
100000368: 39003be1    	strb	w1, [sp, #0xe]
10000036c: 390037e2    	strb	w2, [sp, #0xd]
100000370: 39403fe8    	ldrb	w8, [sp, #0xf]
100000374: 360000e8    	tbz	w8, #0x0, 0x100000390 <__Z20test_logical_complexbbb+0x30>
100000378: 14000001    	b	0x10000037c <__Z20test_logical_complexbbb+0x1c>
10000037c: 39403be8    	ldrb	w8, [sp, #0xe]
100000380: 52800029    	mov	w9, #0x1                ; =1
100000384: b9000be9    	str	w9, [sp, #0x8]
100000388: 370000c8    	tbnz	w8, #0x0, 0x1000003a0 <__Z20test_logical_complexbbb+0x40>
10000038c: 14000001    	b	0x100000390 <__Z20test_logical_complexbbb+0x30>
100000390: 394037e8    	ldrb	w8, [sp, #0xd]
100000394: 52000108    	eor	w8, w8, #0x1
100000398: b9000be8    	str	w8, [sp, #0x8]
10000039c: 14000001    	b	0x1000003a0 <__Z20test_logical_complexbbb+0x40>
1000003a0: b9400be8    	ldr	w8, [sp, #0x8]
1000003a4: 12000100    	and	w0, w8, #0x1
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: 52800008    	mov	w8, #0x0                ; =0
1000003c0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c4: 52800029    	mov	w9, #0x1                ; =1
1000003c8: 12000100    	and	w0, w8, #0x1
1000003cc: 12000121    	and	w1, w9, #0x1
1000003d0: 12000102    	and	w2, w8, #0x1
1000003d4: 97ffffe3    	bl	0x100000360 <__Z20test_logical_complexbbb>
1000003d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003dc: 910083ff    	add	sp, sp, #0x20
1000003e0: d65f03c0    	ret

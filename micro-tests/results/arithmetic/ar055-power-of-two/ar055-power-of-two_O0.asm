
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar055-power-of-two/ar055-power-of-two_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_is_power_of_twoj>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 52800009    	mov	w9, #0x0                ; =0
100000370: b9000be9    	str	w9, [sp, #0x8]
100000374: 34000128    	cbz	w8, 0x100000398 <__Z20test_is_power_of_twoj+0x38>
100000378: 14000001    	b	0x10000037c <__Z20test_is_power_of_twoj+0x1c>
10000037c: b9400fe8    	ldr	w8, [sp, #0xc]
100000380: b9400fe9    	ldr	w9, [sp, #0xc]
100000384: 71000529    	subs	w9, w9, #0x1
100000388: 6a090108    	ands	w8, w8, w9
10000038c: 1a9f17e8    	cset	w8, eq
100000390: b9000be8    	str	w8, [sp, #0x8]
100000394: 14000001    	b	0x100000398 <__Z20test_is_power_of_twoj+0x38>
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: 12000100    	and	w0, w8, #0x1
1000003a0: 910043ff    	add	sp, sp, #0x10
1000003a4: d65f03c0    	ret

00000001000003a8 <_main>:
1000003a8: d10083ff    	sub	sp, sp, #0x20
1000003ac: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b0: 910043fd    	add	x29, sp, #0x10
1000003b4: 52800008    	mov	w8, #0x0                ; =0
1000003b8: b9000be8    	str	w8, [sp, #0x8]
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 52800200    	mov	w0, #0x10               ; =16
1000003c4: 97ffffe7    	bl	0x100000360 <__Z20test_is_power_of_twoj>
1000003c8: b9400be8    	ldr	w8, [sp, #0x8]
1000003cc: 72000009    	ands	w9, w0, #0x1
1000003d0: 1a9f0500    	csinc	w0, w8, wzr, eq
1000003d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d8: 910083ff    	add	sp, sp, #0x20
1000003dc: d65f03c0    	ret

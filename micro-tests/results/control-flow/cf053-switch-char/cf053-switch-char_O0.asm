
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf053-switch-char/cf053-switch-char_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_switch_charc>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39002fe0    	strb	w0, [sp, #0xb]
100000368: 39c02fe8    	ldrsb	w8, [sp, #0xb]
10000036c: b90007e8    	str	w8, [sp, #0x4]
100000370: 71018508    	subs	w8, w8, #0x61
100000374: 54000140    	b.eq	0x10000039c <__Z16test_switch_charc+0x3c>
100000378: 14000001    	b	0x10000037c <__Z16test_switch_charc+0x1c>
10000037c: b94007e8    	ldr	w8, [sp, #0x4]
100000380: 71018908    	subs	w8, w8, #0x62
100000384: 54000120    	b.eq	0x1000003a8 <__Z16test_switch_charc+0x48>
100000388: 14000001    	b	0x10000038c <__Z16test_switch_charc+0x2c>
10000038c: b94007e8    	ldr	w8, [sp, #0x4]
100000390: 71018d08    	subs	w8, w8, #0x63
100000394: 54000100    	b.eq	0x1000003b4 <__Z16test_switch_charc+0x54>
100000398: 1400000a    	b	0x1000003c0 <__Z16test_switch_charc+0x60>
10000039c: 52800028    	mov	w8, #0x1                ; =1
1000003a0: b9000fe8    	str	w8, [sp, #0xc]
1000003a4: 14000009    	b	0x1000003c8 <__Z16test_switch_charc+0x68>
1000003a8: 52800048    	mov	w8, #0x2                ; =2
1000003ac: b9000fe8    	str	w8, [sp, #0xc]
1000003b0: 14000006    	b	0x1000003c8 <__Z16test_switch_charc+0x68>
1000003b4: 52800068    	mov	w8, #0x3                ; =3
1000003b8: b9000fe8    	str	w8, [sp, #0xc]
1000003bc: 14000003    	b	0x1000003c8 <__Z16test_switch_charc+0x68>
1000003c0: b9000fff    	str	wzr, [sp, #0xc]
1000003c4: 14000001    	b	0x1000003c8 <__Z16test_switch_charc+0x68>
1000003c8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003cc: 910043ff    	add	sp, sp, #0x10
1000003d0: d65f03c0    	ret

00000001000003d4 <_main>:
1000003d4: d10083ff    	sub	sp, sp, #0x20
1000003d8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003dc: 910043fd    	add	x29, sp, #0x10
1000003e0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e4: 52800c40    	mov	w0, #0x62               ; =98
1000003e8: 97ffffde    	bl	0x100000360 <__Z16test_switch_charc>
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret

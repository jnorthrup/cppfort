
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf054-switch-enum/cf054-switch-enum_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_switch_enum5Color>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: b90007e8    	str	w8, [sp, #0x4]
100000370: 34000148    	cbz	w8, 0x100000398 <__Z16test_switch_enum5Color+0x38>
100000374: 14000001    	b	0x100000378 <__Z16test_switch_enum5Color+0x18>
100000378: b94007e8    	ldr	w8, [sp, #0x4]
10000037c: 71000508    	subs	w8, w8, #0x1
100000380: 54000120    	b.eq	0x1000003a4 <__Z16test_switch_enum5Color+0x44>
100000384: 14000001    	b	0x100000388 <__Z16test_switch_enum5Color+0x28>
100000388: b94007e8    	ldr	w8, [sp, #0x4]
10000038c: 71000908    	subs	w8, w8, #0x2
100000390: 54000100    	b.eq	0x1000003b0 <__Z16test_switch_enum5Color+0x50>
100000394: 1400000a    	b	0x1000003bc <__Z16test_switch_enum5Color+0x5c>
100000398: 52800148    	mov	w8, #0xa                ; =10
10000039c: b9000fe8    	str	w8, [sp, #0xc]
1000003a0: 14000009    	b	0x1000003c4 <__Z16test_switch_enum5Color+0x64>
1000003a4: 52800288    	mov	w8, #0x14               ; =20
1000003a8: b9000fe8    	str	w8, [sp, #0xc]
1000003ac: 14000006    	b	0x1000003c4 <__Z16test_switch_enum5Color+0x64>
1000003b0: 528003c8    	mov	w8, #0x1e               ; =30
1000003b4: b9000fe8    	str	w8, [sp, #0xc]
1000003b8: 14000003    	b	0x1000003c4 <__Z16test_switch_enum5Color+0x64>
1000003bc: b9000fff    	str	wzr, [sp, #0xc]
1000003c0: 14000001    	b	0x1000003c4 <__Z16test_switch_enum5Color+0x64>
1000003c4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c8: 910043ff    	add	sp, sp, #0x10
1000003cc: d65f03c0    	ret

00000001000003d0 <_main>:
1000003d0: d10083ff    	sub	sp, sp, #0x20
1000003d4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d8: 910043fd    	add	x29, sp, #0x10
1000003dc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e0: 52800020    	mov	w0, #0x1                ; =1
1000003e4: 97ffffdf    	bl	0x100000360 <__Z16test_switch_enum5Color>
1000003e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ec: 910083ff    	add	sp, sp, #0x20
1000003f0: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/control-flow/cf060-switch-large-values/cf060-switch-large-values_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_switch_largei>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: b90007e8    	str	w8, [sp, #0x4]
100000370: 71019108    	subs	w8, w8, #0x64
100000374: 540001c0    	b.eq	0x1000003ac <__Z17test_switch_largei+0x4c>
100000378: 14000001    	b	0x10000037c <__Z17test_switch_largei+0x1c>
10000037c: b94007e8    	ldr	w8, [sp, #0x4]
100000380: 71032108    	subs	w8, w8, #0xc8
100000384: 540001a0    	b.eq	0x1000003b8 <__Z17test_switch_largei+0x58>
100000388: 14000001    	b	0x10000038c <__Z17test_switch_largei+0x2c>
10000038c: b94007e8    	ldr	w8, [sp, #0x4]
100000390: 7104b108    	subs	w8, w8, #0x12c
100000394: 54000180    	b.eq	0x1000003c4 <__Z17test_switch_largei+0x64>
100000398: 14000001    	b	0x10000039c <__Z17test_switch_largei+0x3c>
10000039c: b94007e8    	ldr	w8, [sp, #0x4]
1000003a0: 710fa108    	subs	w8, w8, #0x3e8
1000003a4: 54000160    	b.eq	0x1000003d0 <__Z17test_switch_largei+0x70>
1000003a8: 1400000d    	b	0x1000003dc <__Z17test_switch_largei+0x7c>
1000003ac: 52800028    	mov	w8, #0x1                ; =1
1000003b0: b9000fe8    	str	w8, [sp, #0xc]
1000003b4: 1400000c    	b	0x1000003e4 <__Z17test_switch_largei+0x84>
1000003b8: 52800048    	mov	w8, #0x2                ; =2
1000003bc: b9000fe8    	str	w8, [sp, #0xc]
1000003c0: 14000009    	b	0x1000003e4 <__Z17test_switch_largei+0x84>
1000003c4: 52800068    	mov	w8, #0x3                ; =3
1000003c8: b9000fe8    	str	w8, [sp, #0xc]
1000003cc: 14000006    	b	0x1000003e4 <__Z17test_switch_largei+0x84>
1000003d0: 52800088    	mov	w8, #0x4                ; =4
1000003d4: b9000fe8    	str	w8, [sp, #0xc]
1000003d8: 14000003    	b	0x1000003e4 <__Z17test_switch_largei+0x84>
1000003dc: b9000fff    	str	wzr, [sp, #0xc]
1000003e0: 14000001    	b	0x1000003e4 <__Z17test_switch_largei+0x84>
1000003e4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003e8: 910043ff    	add	sp, sp, #0x10
1000003ec: d65f03c0    	ret

00000001000003f0 <_main>:
1000003f0: d10083ff    	sub	sp, sp, #0x20
1000003f4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003f8: 910043fd    	add	x29, sp, #0x10
1000003fc: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000400: 52802580    	mov	w0, #0x12c              ; =300
100000404: 97ffffd7    	bl	0x100000360 <__Z17test_switch_largei>
100000408: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000040c: 910083ff    	add	sp, sp, #0x20
100000410: d65f03c0    	ret

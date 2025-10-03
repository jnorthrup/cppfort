
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf058-switch-return-in-case/cf058-switch-return-in-case_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_switch_returni>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: b90007e8    	str	w8, [sp, #0x4]
100000370: 71000508    	subs	w8, w8, #0x1
100000374: 54000140    	b.eq	0x10000039c <__Z18test_switch_returni+0x3c>
100000378: 14000001    	b	0x10000037c <__Z18test_switch_returni+0x1c>
10000037c: b94007e8    	ldr	w8, [sp, #0x4]
100000380: 71000908    	subs	w8, w8, #0x2
100000384: 54000120    	b.eq	0x1000003a8 <__Z18test_switch_returni+0x48>
100000388: 14000001    	b	0x10000038c <__Z18test_switch_returni+0x2c>
10000038c: b94007e8    	ldr	w8, [sp, #0x4]
100000390: 71000d08    	subs	w8, w8, #0x3
100000394: 54000100    	b.eq	0x1000003b4 <__Z18test_switch_returni+0x54>
100000398: 1400000a    	b	0x1000003c0 <__Z18test_switch_returni+0x60>
10000039c: 52800148    	mov	w8, #0xa                ; =10
1000003a0: b9000fe8    	str	w8, [sp, #0xc]
1000003a4: 14000009    	b	0x1000003c8 <__Z18test_switch_returni+0x68>
1000003a8: 52800288    	mov	w8, #0x14               ; =20
1000003ac: b9000fe8    	str	w8, [sp, #0xc]
1000003b0: 14000006    	b	0x1000003c8 <__Z18test_switch_returni+0x68>
1000003b4: 528003c8    	mov	w8, #0x1e               ; =30
1000003b8: b9000fe8    	str	w8, [sp, #0xc]
1000003bc: 14000003    	b	0x1000003c8 <__Z18test_switch_returni+0x68>
1000003c0: b9000fff    	str	wzr, [sp, #0xc]
1000003c4: 14000001    	b	0x1000003c8 <__Z18test_switch_returni+0x68>
1000003c8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003cc: 910043ff    	add	sp, sp, #0x10
1000003d0: d65f03c0    	ret

00000001000003d4 <_main>:
1000003d4: d10083ff    	sub	sp, sp, #0x20
1000003d8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003dc: 910043fd    	add	x29, sp, #0x10
1000003e0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e4: 52800040    	mov	w0, #0x2                ; =2
1000003e8: 97ffffde    	bl	0x100000360 <__Z18test_switch_returni>
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret

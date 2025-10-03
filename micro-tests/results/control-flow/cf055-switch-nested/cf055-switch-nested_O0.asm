
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf055-switch-nested/cf055-switch-nested_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_nested_switchii>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: b9001be0    	str	w0, [sp, #0x18]
100000368: b90017e1    	str	w1, [sp, #0x14]
10000036c: b9401be8    	ldr	w8, [sp, #0x18]
100000370: b90013e8    	str	w8, [sp, #0x10]
100000374: 71000508    	subs	w8, w8, #0x1
100000378: 540000c0    	b.eq	0x100000390 <__Z18test_nested_switchii+0x30>
10000037c: 14000001    	b	0x100000380 <__Z18test_nested_switchii+0x20>
100000380: b94013e8    	ldr	w8, [sp, #0x10]
100000384: 71000908    	subs	w8, w8, #0x2
100000388: 54000280    	b.eq	0x1000003d8 <__Z18test_nested_switchii+0x78>
10000038c: 14000016    	b	0x1000003e4 <__Z18test_nested_switchii+0x84>
100000390: b94017e8    	ldr	w8, [sp, #0x14]
100000394: b9000fe8    	str	w8, [sp, #0xc]
100000398: 71000508    	subs	w8, w8, #0x1
10000039c: 540000c0    	b.eq	0x1000003b4 <__Z18test_nested_switchii+0x54>
1000003a0: 14000001    	b	0x1000003a4 <__Z18test_nested_switchii+0x44>
1000003a4: b9400fe8    	ldr	w8, [sp, #0xc]
1000003a8: 71000908    	subs	w8, w8, #0x2
1000003ac: 540000a0    	b.eq	0x1000003c0 <__Z18test_nested_switchii+0x60>
1000003b0: 14000007    	b	0x1000003cc <__Z18test_nested_switchii+0x6c>
1000003b4: 52800168    	mov	w8, #0xb                ; =11
1000003b8: b9001fe8    	str	w8, [sp, #0x1c]
1000003bc: 1400000c    	b	0x1000003ec <__Z18test_nested_switchii+0x8c>
1000003c0: 52800188    	mov	w8, #0xc                ; =12
1000003c4: b9001fe8    	str	w8, [sp, #0x1c]
1000003c8: 14000009    	b	0x1000003ec <__Z18test_nested_switchii+0x8c>
1000003cc: 52800148    	mov	w8, #0xa                ; =10
1000003d0: b9001fe8    	str	w8, [sp, #0x1c]
1000003d4: 14000006    	b	0x1000003ec <__Z18test_nested_switchii+0x8c>
1000003d8: 52800288    	mov	w8, #0x14               ; =20
1000003dc: b9001fe8    	str	w8, [sp, #0x1c]
1000003e0: 14000003    	b	0x1000003ec <__Z18test_nested_switchii+0x8c>
1000003e4: b9001fff    	str	wzr, [sp, #0x1c]
1000003e8: 14000001    	b	0x1000003ec <__Z18test_nested_switchii+0x8c>
1000003ec: b9401fe0    	ldr	w0, [sp, #0x1c]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret

00000001000003f8 <_main>:
1000003f8: d10083ff    	sub	sp, sp, #0x20
1000003fc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000400: 910043fd    	add	x29, sp, #0x10
100000404: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000408: 52800020    	mov	w0, #0x1                ; =1
10000040c: 52800041    	mov	w1, #0x2                ; =2
100000410: 97ffffd4    	bl	0x100000360 <__Z18test_nested_switchii>
100000414: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000418: 910083ff    	add	sp, sp, #0x20
10000041c: d65f03c0    	ret

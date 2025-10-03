
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf066-goto-multiple-labels/cf066-goto-multiple-labels_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_multiple_gotosi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 71000508    	subs	w8, w8, #0x1
100000370: 54000061    	b.ne	0x10000037c <__Z19test_multiple_gotosi+0x1c>
100000374: 14000001    	b	0x100000378 <__Z19test_multiple_gotosi+0x18>
100000378: 1400000d    	b	0x1000003ac <__Z19test_multiple_gotosi+0x4c>
10000037c: b9400be8    	ldr	w8, [sp, #0x8]
100000380: 71000908    	subs	w8, w8, #0x2
100000384: 54000061    	b.ne	0x100000390 <__Z19test_multiple_gotosi+0x30>
100000388: 14000001    	b	0x10000038c <__Z19test_multiple_gotosi+0x2c>
10000038c: 1400000b    	b	0x1000003b8 <__Z19test_multiple_gotosi+0x58>
100000390: b9400be8    	ldr	w8, [sp, #0x8]
100000394: 71000d08    	subs	w8, w8, #0x3
100000398: 54000061    	b.ne	0x1000003a4 <__Z19test_multiple_gotosi+0x44>
10000039c: 14000001    	b	0x1000003a0 <__Z19test_multiple_gotosi+0x40>
1000003a0: 14000009    	b	0x1000003c4 <__Z19test_multiple_gotosi+0x64>
1000003a4: b9000fff    	str	wzr, [sp, #0xc]
1000003a8: 1400000a    	b	0x1000003d0 <__Z19test_multiple_gotosi+0x70>
1000003ac: 52800148    	mov	w8, #0xa                ; =10
1000003b0: b9000fe8    	str	w8, [sp, #0xc]
1000003b4: 14000007    	b	0x1000003d0 <__Z19test_multiple_gotosi+0x70>
1000003b8: 52800288    	mov	w8, #0x14               ; =20
1000003bc: b9000fe8    	str	w8, [sp, #0xc]
1000003c0: 14000004    	b	0x1000003d0 <__Z19test_multiple_gotosi+0x70>
1000003c4: 528003c8    	mov	w8, #0x1e               ; =30
1000003c8: b9000fe8    	str	w8, [sp, #0xc]
1000003cc: 14000001    	b	0x1000003d0 <__Z19test_multiple_gotosi+0x70>
1000003d0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003d4: 910043ff    	add	sp, sp, #0x10
1000003d8: d65f03c0    	ret

00000001000003dc <_main>:
1000003dc: d10083ff    	sub	sp, sp, #0x20
1000003e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e4: 910043fd    	add	x29, sp, #0x10
1000003e8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ec: 52800040    	mov	w0, #0x2                ; =2
1000003f0: 97ffffdc    	bl	0x100000360 <__Z19test_multiple_gotosi>
1000003f4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f8: 910083ff    	add	sp, sp, #0x20
1000003fc: d65f03c0    	ret

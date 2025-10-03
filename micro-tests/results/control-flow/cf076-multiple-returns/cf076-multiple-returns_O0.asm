
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf076-multiple-returns/cf076-multiple-returns_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_multiple_returnsi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 36f800a8    	tbz	w8, #0x1f, 0x100000380 <__Z21test_multiple_returnsi+0x20>
100000370: 14000001    	b	0x100000374 <__Z21test_multiple_returnsi+0x14>
100000374: 12800008    	mov	w8, #-0x1               ; =-1
100000378: b9000fe8    	str	w8, [sp, #0xc]
10000037c: 14000017    	b	0x1000003d8 <__Z21test_multiple_returnsi+0x78>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 35000088    	cbnz	w8, 0x100000394 <__Z21test_multiple_returnsi+0x34>
100000388: 14000001    	b	0x10000038c <__Z21test_multiple_returnsi+0x2c>
10000038c: b9000fff    	str	wzr, [sp, #0xc]
100000390: 14000012    	b	0x1000003d8 <__Z21test_multiple_returnsi+0x78>
100000394: b9400be8    	ldr	w8, [sp, #0x8]
100000398: 71002908    	subs	w8, w8, #0xa
10000039c: 540000aa    	b.ge	0x1000003b0 <__Z21test_multiple_returnsi+0x50>
1000003a0: 14000001    	b	0x1000003a4 <__Z21test_multiple_returnsi+0x44>
1000003a4: 52800028    	mov	w8, #0x1                ; =1
1000003a8: b9000fe8    	str	w8, [sp, #0xc]
1000003ac: 1400000b    	b	0x1000003d8 <__Z21test_multiple_returnsi+0x78>
1000003b0: b9400be8    	ldr	w8, [sp, #0x8]
1000003b4: 71019108    	subs	w8, w8, #0x64
1000003b8: 540000aa    	b.ge	0x1000003cc <__Z21test_multiple_returnsi+0x6c>
1000003bc: 14000001    	b	0x1000003c0 <__Z21test_multiple_returnsi+0x60>
1000003c0: 52800048    	mov	w8, #0x2                ; =2
1000003c4: b9000fe8    	str	w8, [sp, #0xc]
1000003c8: 14000004    	b	0x1000003d8 <__Z21test_multiple_returnsi+0x78>
1000003cc: 52800068    	mov	w8, #0x3                ; =3
1000003d0: b9000fe8    	str	w8, [sp, #0xc]
1000003d4: 14000001    	b	0x1000003d8 <__Z21test_multiple_returnsi+0x78>
1000003d8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003dc: 910043ff    	add	sp, sp, #0x10
1000003e0: d65f03c0    	ret

00000001000003e4 <_main>:
1000003e4: d10083ff    	sub	sp, sp, #0x20
1000003e8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003ec: 910043fd    	add	x29, sp, #0x10
1000003f0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003f4: 52800640    	mov	w0, #0x32               ; =50
1000003f8: 97ffffda    	bl	0x100000360 <__Z21test_multiple_returnsi>
1000003fc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000400: 910083ff    	add	sp, sp, #0x20
100000404: d65f03c0    	ret

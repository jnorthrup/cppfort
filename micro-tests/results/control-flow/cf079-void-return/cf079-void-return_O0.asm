
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf079-void-return/cf079-void-return_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_void_returniRi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: f90003e1    	str	x1, [sp]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: 36f800c8    	tbz	w8, #0x1f, 0x100000388 <__Z16test_void_returniRi+0x28>
100000374: 14000001    	b	0x100000378 <__Z16test_void_returniRi+0x18>
100000378: f94003e9    	ldr	x9, [sp]
10000037c: 12800008    	mov	w8, #-0x1               ; =-1
100000380: b9000128    	str	w8, [x9]
100000384: 1400000c    	b	0x1000003b4 <__Z16test_void_returniRi+0x54>
100000388: b9400fe8    	ldr	w8, [sp, #0xc]
10000038c: 350000a8    	cbnz	w8, 0x1000003a0 <__Z16test_void_returniRi+0x40>
100000390: 14000001    	b	0x100000394 <__Z16test_void_returniRi+0x34>
100000394: f94003e8    	ldr	x8, [sp]
100000398: b900011f    	str	wzr, [x8]
10000039c: 14000006    	b	0x1000003b4 <__Z16test_void_returniRi+0x54>
1000003a0: b9400fe8    	ldr	w8, [sp, #0xc]
1000003a4: 531f7908    	lsl	w8, w8, #1
1000003a8: f94003e9    	ldr	x9, [sp]
1000003ac: b9000128    	str	w8, [x9]
1000003b0: 14000001    	b	0x1000003b4 <__Z16test_void_returniRi+0x54>
1000003b4: 910043ff    	add	sp, sp, #0x10
1000003b8: d65f03c0    	ret

00000001000003bc <_main>:
1000003bc: d10083ff    	sub	sp, sp, #0x20
1000003c0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c4: 910043fd    	add	x29, sp, #0x10
1000003c8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003cc: 910023e1    	add	x1, sp, #0x8
1000003d0: b9000bff    	str	wzr, [sp, #0x8]
1000003d4: 52800140    	mov	w0, #0xa                ; =10
1000003d8: 97ffffe2    	bl	0x100000360 <__Z16test_void_returniRi>
1000003dc: b9400be0    	ldr	w0, [sp, #0x8]
1000003e0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e4: 910083ff    	add	sp, sp, #0x20
1000003e8: d65f03c0    	ret

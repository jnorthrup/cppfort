
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf004-if-else-if/cf004-if-else-if_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_if_else_ifi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 36f800a8    	tbz	w8, #0x1f, 0x100000380 <__Z15test_if_else_ifi+0x20>
100000370: 14000001    	b	0x100000374 <__Z15test_if_else_ifi+0x14>
100000374: 12800008    	mov	w8, #-0x1               ; =-1
100000378: b9000fe8    	str	w8, [sp, #0xc]
10000037c: 14000010    	b	0x1000003bc <__Z15test_if_else_ifi+0x5c>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 35000088    	cbnz	w8, 0x100000394 <__Z15test_if_else_ifi+0x34>
100000388: 14000001    	b	0x10000038c <__Z15test_if_else_ifi+0x2c>
10000038c: b9000fff    	str	wzr, [sp, #0xc]
100000390: 1400000b    	b	0x1000003bc <__Z15test_if_else_ifi+0x5c>
100000394: b9400be8    	ldr	w8, [sp, #0x8]
100000398: 71002908    	subs	w8, w8, #0xa
10000039c: 540000aa    	b.ge	0x1000003b0 <__Z15test_if_else_ifi+0x50>
1000003a0: 14000001    	b	0x1000003a4 <__Z15test_if_else_ifi+0x44>
1000003a4: 52800028    	mov	w8, #0x1                ; =1
1000003a8: b9000fe8    	str	w8, [sp, #0xc]
1000003ac: 14000004    	b	0x1000003bc <__Z15test_if_else_ifi+0x5c>
1000003b0: 52800048    	mov	w8, #0x2                ; =2
1000003b4: b9000fe8    	str	w8, [sp, #0xc]
1000003b8: 14000001    	b	0x1000003bc <__Z15test_if_else_ifi+0x5c>
1000003bc: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c0: 910043ff    	add	sp, sp, #0x10
1000003c4: d65f03c0    	ret

00000001000003c8 <_main>:
1000003c8: d10083ff    	sub	sp, sp, #0x20
1000003cc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d0: 910043fd    	add	x29, sp, #0x10
1000003d4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d8: 528000a0    	mov	w0, #0x5                ; =5
1000003dc: 97ffffe1    	bl	0x100000360 <__Z15test_if_else_ifi>
1000003e0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e4: 910083ff    	add	sp, sp, #0x20
1000003e8: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/control-flow/cf047-do-while-decrement/cf047-do-while-decrement_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_do_while_decrementv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: 52800148    	mov	w8, #0xa                ; =10
10000036c: b9000be8    	str	w8, [sp, #0x8]
100000370: 14000001    	b	0x100000374 <__Z23test_do_while_decrementv+0x14>
100000374: b9400be9    	ldr	w9, [sp, #0x8]
100000378: b9400fe8    	ldr	w8, [sp, #0xc]
10000037c: 0b090108    	add	w8, w8, w9
100000380: b9000fe8    	str	w8, [sp, #0xc]
100000384: b9400be8    	ldr	w8, [sp, #0x8]
100000388: 71000508    	subs	w8, w8, #0x1
10000038c: b9000be8    	str	w8, [sp, #0x8]
100000390: 14000001    	b	0x100000394 <__Z23test_do_while_decrementv+0x34>
100000394: b9400be8    	ldr	w8, [sp, #0x8]
100000398: 71000108    	subs	w8, w8, #0x0
10000039c: 54fffecc    	b.gt	0x100000374 <__Z23test_do_while_decrementv+0x14>
1000003a0: 14000001    	b	0x1000003a4 <__Z23test_do_while_decrementv+0x44>
1000003a4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 97ffffe8    	bl	0x100000360 <__Z23test_do_while_decrementv>
1000003c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c8: 910083ff    	add	sp, sp, #0x20
1000003cc: d65f03c0    	ret

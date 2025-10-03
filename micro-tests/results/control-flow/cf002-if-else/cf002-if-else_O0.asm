
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf002-if-else/cf002-if-else_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z12test_if_elsei>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 71000108    	subs	w8, w8, #0x0
100000370: 540000ad    	b.le	0x100000384 <__Z12test_if_elsei+0x24>
100000374: 14000001    	b	0x100000378 <__Z12test_if_elsei+0x18>
100000378: 52800028    	mov	w8, #0x1                ; =1
10000037c: b9000fe8    	str	w8, [sp, #0xc]
100000380: 14000004    	b	0x100000390 <__Z12test_if_elsei+0x30>
100000384: 12800008    	mov	w8, #-0x1               ; =-1
100000388: b9000fe8    	str	w8, [sp, #0xc]
10000038c: 14000001    	b	0x100000390 <__Z12test_if_elsei+0x30>
100000390: b9400fe0    	ldr	w0, [sp, #0xc]
100000394: 910043ff    	add	sp, sp, #0x10
100000398: d65f03c0    	ret

000000010000039c <_main>:
10000039c: d10083ff    	sub	sp, sp, #0x20
1000003a0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a4: 910043fd    	add	x29, sp, #0x10
1000003a8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ac: 528000a0    	mov	w0, #0x5                ; =5
1000003b0: 97ffffec    	bl	0x100000360 <__Z12test_if_elsei>
1000003b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b8: 910083ff    	add	sp, sp, #0x20
1000003bc: d65f03c0    	ret

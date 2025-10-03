
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf001-simple-if/cf001-simple-if_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_simple_ifi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 71000108    	subs	w8, w8, #0x0
100000370: 540000ad    	b.le	0x100000384 <__Z14test_simple_ifi+0x24>
100000374: 14000001    	b	0x100000378 <__Z14test_simple_ifi+0x18>
100000378: 52800028    	mov	w8, #0x1                ; =1
10000037c: b9000fe8    	str	w8, [sp, #0xc]
100000380: 14000003    	b	0x10000038c <__Z14test_simple_ifi+0x2c>
100000384: b9000fff    	str	wzr, [sp, #0xc]
100000388: 14000001    	b	0x10000038c <__Z14test_simple_ifi+0x2c>
10000038c: b9400fe0    	ldr	w0, [sp, #0xc]
100000390: 910043ff    	add	sp, sp, #0x10
100000394: d65f03c0    	ret

0000000100000398 <_main>:
100000398: d10083ff    	sub	sp, sp, #0x20
10000039c: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a0: 910043fd    	add	x29, sp, #0x10
1000003a4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a8: 528000a0    	mov	w0, #0x5                ; =5
1000003ac: 97ffffed    	bl	0x100000360 <__Z14test_simple_ifi>
1000003b0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b4: 910083ff    	add	sp, sp, #0x20
1000003b8: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/control-flow/cf099-state-machine/cf099-state-machine_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_state_machinei>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: b9001fe0    	str	w0, [sp, #0x1c]
100000368: b9001bff    	str	wzr, [sp, #0x18]
10000036c: b90017ff    	str	wzr, [sp, #0x14]
100000370: 14000001    	b	0x100000374 <__Z18test_state_machinei+0x14>
100000374: b9401be9    	ldr	w9, [sp, #0x18]
100000378: 52800008    	mov	w8, #0x0                ; =0
10000037c: 71000929    	subs	w9, w9, #0x2
100000380: b90013e8    	str	w8, [sp, #0x10]
100000384: 540000e0    	b.eq	0x1000003a0 <__Z18test_state_machinei+0x40>
100000388: 14000001    	b	0x10000038c <__Z18test_state_machinei+0x2c>
10000038c: b9401be8    	ldr	w8, [sp, #0x18]
100000390: 71000d08    	subs	w8, w8, #0x3
100000394: 1a9f07e8    	cset	w8, ne
100000398: b90013e8    	str	w8, [sp, #0x10]
10000039c: 14000001    	b	0x1000003a0 <__Z18test_state_machinei+0x40>
1000003a0: b94013e8    	ldr	w8, [sp, #0x10]
1000003a4: 360003e8    	tbz	w8, #0x0, 0x100000420 <__Z18test_state_machinei+0xc0>
1000003a8: 14000001    	b	0x1000003ac <__Z18test_state_machinei+0x4c>
1000003ac: b9401be8    	ldr	w8, [sp, #0x18]
1000003b0: b9000fe8    	str	w8, [sp, #0xc]
1000003b4: 340000c8    	cbz	w8, 0x1000003cc <__Z18test_state_machinei+0x6c>
1000003b8: 14000001    	b	0x1000003bc <__Z18test_state_machinei+0x5c>
1000003bc: b9400fe8    	ldr	w8, [sp, #0xc]
1000003c0: 71000508    	subs	w8, w8, #0x1
1000003c4: 540001a0    	b.eq	0x1000003f8 <__Z18test_state_machinei+0x98>
1000003c8: 14000012    	b	0x100000410 <__Z18test_state_machinei+0xb0>
1000003cc: b9401fe8    	ldr	w8, [sp, #0x1c]
1000003d0: 71000108    	subs	w8, w8, #0x0
1000003d4: 540000ad    	b.le	0x1000003e8 <__Z18test_state_machinei+0x88>
1000003d8: 14000001    	b	0x1000003dc <__Z18test_state_machinei+0x7c>
1000003dc: 52800028    	mov	w8, #0x1                ; =1
1000003e0: b9001be8    	str	w8, [sp, #0x18]
1000003e4: 14000004    	b	0x1000003f4 <__Z18test_state_machinei+0x94>
1000003e8: 52800068    	mov	w8, #0x3                ; =3
1000003ec: b9001be8    	str	w8, [sp, #0x18]
1000003f0: 14000001    	b	0x1000003f4 <__Z18test_state_machinei+0x94>
1000003f4: 1400000a    	b	0x10000041c <__Z18test_state_machinei+0xbc>
1000003f8: b9401fe8    	ldr	w8, [sp, #0x1c]
1000003fc: 531f7908    	lsl	w8, w8, #1
100000400: b90017e8    	str	w8, [sp, #0x14]
100000404: 52800048    	mov	w8, #0x2                ; =2
100000408: b9001be8    	str	w8, [sp, #0x18]
10000040c: 14000004    	b	0x10000041c <__Z18test_state_machinei+0xbc>
100000410: 52800068    	mov	w8, #0x3                ; =3
100000414: b9001be8    	str	w8, [sp, #0x18]
100000418: 14000001    	b	0x10000041c <__Z18test_state_machinei+0xbc>
10000041c: 17ffffd6    	b	0x100000374 <__Z18test_state_machinei+0x14>
100000420: b9401be8    	ldr	w8, [sp, #0x18]
100000424: 71000908    	subs	w8, w8, #0x2
100000428: 540000a1    	b.ne	0x10000043c <__Z18test_state_machinei+0xdc>
10000042c: 14000001    	b	0x100000430 <__Z18test_state_machinei+0xd0>
100000430: b94017e8    	ldr	w8, [sp, #0x14]
100000434: b9000be8    	str	w8, [sp, #0x8]
100000438: 14000004    	b	0x100000448 <__Z18test_state_machinei+0xe8>
10000043c: 12800008    	mov	w8, #-0x1               ; =-1
100000440: b9000be8    	str	w8, [sp, #0x8]
100000444: 14000001    	b	0x100000448 <__Z18test_state_machinei+0xe8>
100000448: b9400be0    	ldr	w0, [sp, #0x8]
10000044c: 910083ff    	add	sp, sp, #0x20
100000450: d65f03c0    	ret

0000000100000454 <_main>:
100000454: d10083ff    	sub	sp, sp, #0x20
100000458: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000045c: 910043fd    	add	x29, sp, #0x10
100000460: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000464: 528002a0    	mov	w0, #0x15               ; =21
100000468: 97ffffbe    	bl	0x100000360 <__Z18test_state_machinei>
10000046c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000470: 910083ff    	add	sp, sp, #0x20
100000474: d65f03c0    	ret

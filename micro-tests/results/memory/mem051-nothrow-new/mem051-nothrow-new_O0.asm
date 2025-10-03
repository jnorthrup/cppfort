
/Users/jim/work/cppfort/micro-tests/results/memory/mem051-nothrow-new/mem051-nothrow-new_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z16test_nothrow_newv>:
100000448: d10103ff    	sub	sp, sp, #0x40
10000044c: a9037bfd    	stp	x29, x30, [sp, #0x30]
100000450: 9100c3fd    	add	x29, sp, #0x30
100000454: d2800080    	mov	x0, #0x4                ; =4
100000458: 90000021    	adrp	x1, 0x100004000 <__ZnwmRKSt9nothrow_t+0x100004000>
10000045c: f9400021    	ldr	x1, [x1]
100000460: 94000034    	bl	0x100000530 <__ZnwmRKSt9nothrow_t+0x100000530>
100000464: f90007e0    	str	x0, [sp, #0x8]
100000468: d2800008    	mov	x8, #0x0                ; =0
10000046c: 52800009    	mov	w9, #0x0                ; =0
100000470: 12000129    	and	w9, w9, #0x1
100000474: 12000129    	and	w9, w9, #0x1
100000478: 381ef3a9    	sturb	w9, [x29, #-0x11]
10000047c: f9000be8    	str	x8, [sp, #0x10]
100000480: b4000160    	cbz	x0, 0x1000004ac <__Z16test_nothrow_newv+0x64>
100000484: 14000001    	b	0x100000488 <__Z16test_nothrow_newv+0x40>
100000488: f94007e8    	ldr	x8, [sp, #0x8]
10000048c: 52800029    	mov	w9, #0x1                ; =1
100000490: 12000129    	and	w9, w9, #0x1
100000494: 12000129    	and	w9, w9, #0x1
100000498: 381ef3a9    	sturb	w9, [x29, #-0x11]
10000049c: 52800549    	mov	w9, #0x2a               ; =42
1000004a0: b9000109    	str	w9, [x8]
1000004a4: f9000be8    	str	x8, [sp, #0x10]
1000004a8: 14000001    	b	0x1000004ac <__Z16test_nothrow_newv+0x64>
1000004ac: f9400be8    	ldr	x8, [sp, #0x10]
1000004b0: f81f03a8    	stur	x8, [x29, #-0x10]
1000004b4: f85f03a8    	ldur	x8, [x29, #-0x10]
1000004b8: b50000a8    	cbnz	x8, 0x1000004cc <__Z16test_nothrow_newv+0x84>
1000004bc: 14000001    	b	0x1000004c0 <__Z16test_nothrow_newv+0x78>
1000004c0: 12800008    	mov	w8, #-0x1               ; =-1
1000004c4: b81fc3a8    	stur	w8, [x29, #-0x4]
1000004c8: 1400000e    	b	0x100000500 <__Z16test_nothrow_newv+0xb8>
1000004cc: f85f03a8    	ldur	x8, [x29, #-0x10]
1000004d0: b9400108    	ldr	w8, [x8]
1000004d4: b9001be8    	str	w8, [sp, #0x18]
1000004d8: f85f03a8    	ldur	x8, [x29, #-0x10]
1000004dc: f90003e8    	str	x8, [sp]
1000004e0: b40000a8    	cbz	x8, 0x1000004f4 <__Z16test_nothrow_newv+0xac>
1000004e4: 14000001    	b	0x1000004e8 <__Z16test_nothrow_newv+0xa0>
1000004e8: f94003e0    	ldr	x0, [sp]
1000004ec: 94000014    	bl	0x10000053c <__ZnwmRKSt9nothrow_t+0x10000053c>
1000004f0: 14000001    	b	0x1000004f4 <__Z16test_nothrow_newv+0xac>
1000004f4: b9401be8    	ldr	w8, [sp, #0x18]
1000004f8: b81fc3a8    	stur	w8, [x29, #-0x4]
1000004fc: 14000001    	b	0x100000500 <__Z16test_nothrow_newv+0xb8>
100000500: b85fc3a0    	ldur	w0, [x29, #-0x4]
100000504: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000508: 910103ff    	add	sp, sp, #0x40
10000050c: d65f03c0    	ret

0000000100000510 <_main>:
100000510: d10083ff    	sub	sp, sp, #0x20
100000514: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000518: 910043fd    	add	x29, sp, #0x10
10000051c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000520: 97ffffca    	bl	0x100000448 <__Z16test_nothrow_newv>
100000524: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000528: 910083ff    	add	sp, sp, #0x20
10000052c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000530 <__stubs>:
100000530: 90000030    	adrp	x16, 0x100004000 <__ZnwmRKSt9nothrow_t+0x100004000>
100000534: f9400610    	ldr	x16, [x16, #0x8]
100000538: d61f0200    	br	x16
10000053c: 90000030    	adrp	x16, 0x100004000 <__ZnwmRKSt9nothrow_t+0x100004000>
100000540: f9400a10    	ldr	x16, [x16, #0x10]
100000544: d61f0200    	br	x16

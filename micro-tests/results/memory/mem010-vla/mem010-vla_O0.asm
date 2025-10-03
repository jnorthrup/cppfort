
/Users/jim/work/cppfort/micro-tests/results/memory/mem010-vla/mem010-vla_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000448 <__Z8test_vlai>:
100000448: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000044c: 910003fd    	mov	x29, sp
100000450: d10143ff    	sub	sp, sp, #0x50
100000454: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000458: f9400108    	ldr	x8, [x8]
10000045c: f9400108    	ldr	x8, [x8]
100000460: f81f83a8    	stur	x8, [x29, #-0x8]
100000464: b81f43a0    	stur	w0, [x29, #-0xc]
100000468: b85f43a8    	ldur	w8, [x29, #-0xc]
10000046c: 910003e9    	mov	x9, sp
100000470: f81e83a9    	stur	x9, [x29, #-0x18]
100000474: d37ef509    	lsl	x9, x8, #2
100000478: f81c83a9    	stur	x9, [x29, #-0x38]
10000047c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000480: f9400610    	ldr	x16, [x16, #0x8]
100000484: d63f0200    	blr	x16
100000488: f85c83a9    	ldur	x9, [x29, #-0x38]
10000048c: 91003d29    	add	x9, x9, #0xf
100000490: 927c792a    	and	x10, x9, #0x7fffffff0
100000494: 910003e9    	mov	x9, sp
100000498: eb0a0129    	subs	x9, x9, x10
10000049c: 9100013f    	mov	sp, x9
1000004a0: f81d03a9    	stur	x9, [x29, #-0x30]
1000004a4: f81e03a8    	stur	x8, [x29, #-0x20]
1000004a8: 52800008    	mov	w8, #0x0                ; =0
1000004ac: b81dc3a8    	stur	w8, [x29, #-0x24]
1000004b0: 14000001    	b	0x1000004b4 <__Z8test_vlai+0x6c>
1000004b4: b85dc3a8    	ldur	w8, [x29, #-0x24]
1000004b8: b85f43a9    	ldur	w9, [x29, #-0xc]
1000004bc: 6b090108    	subs	w8, w8, w9
1000004c0: 5400016a    	b.ge	0x1000004ec <__Z8test_vlai+0xa4>
1000004c4: 14000001    	b	0x1000004c8 <__Z8test_vlai+0x80>
1000004c8: f85d03a9    	ldur	x9, [x29, #-0x30]
1000004cc: b89dc3aa    	ldursw	x10, [x29, #-0x24]
1000004d0: aa0a03e8    	mov	x8, x10
1000004d4: b82a7928    	str	w8, [x9, x10, lsl #2]
1000004d8: 14000001    	b	0x1000004dc <__Z8test_vlai+0x94>
1000004dc: b85dc3a8    	ldur	w8, [x29, #-0x24]
1000004e0: 11000508    	add	w8, w8, #0x1
1000004e4: b81dc3a8    	stur	w8, [x29, #-0x24]
1000004e8: 17fffff3    	b	0x1000004b4 <__Z8test_vlai+0x6c>
1000004ec: f85d03a8    	ldur	x8, [x29, #-0x30]
1000004f0: b85f43a9    	ldur	w9, [x29, #-0xc]
1000004f4: 71000529    	subs	w9, w9, #0x1
1000004f8: b869d908    	ldr	w8, [x8, w9, sxtw #2]
1000004fc: b81bc3a8    	stur	w8, [x29, #-0x44]
100000500: f85e83a8    	ldur	x8, [x29, #-0x18]
100000504: f81c03a8    	stur	x8, [x29, #-0x40]
100000508: f85f83a9    	ldur	x9, [x29, #-0x8]
10000050c: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000510: f9400108    	ldr	x8, [x8]
100000514: f9400108    	ldr	x8, [x8]
100000518: eb090108    	subs	x8, x8, x9
10000051c: 54000041    	b.ne	0x100000524 <__Z8test_vlai+0xdc>
100000520: 14000002    	b	0x100000528 <__Z8test_vlai+0xe0>
100000524: 94000010    	bl	0x100000564 <___stack_chk_guard+0x100000564>
100000528: b85bc3a0    	ldur	w0, [x29, #-0x44]
10000052c: f85c03a8    	ldur	x8, [x29, #-0x40]
100000530: 9100011f    	mov	sp, x8
100000534: 910003bf    	mov	sp, x29
100000538: a8c17bfd    	ldp	x29, x30, [sp], #0x10
10000053c: d65f03c0    	ret

0000000100000540 <_main>:
100000540: d10083ff    	sub	sp, sp, #0x20
100000544: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000548: 910043fd    	add	x29, sp, #0x10
10000054c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000550: 528000a0    	mov	w0, #0x5                ; =5
100000554: 97ffffbd    	bl	0x100000448 <__Z8test_vlai>
100000558: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000055c: 910083ff    	add	sp, sp, #0x20
100000560: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000564 <__stubs>:
100000564: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000568: f9400a10    	ldr	x16, [x16, #0x10]
10000056c: d61f0200    	br	x16

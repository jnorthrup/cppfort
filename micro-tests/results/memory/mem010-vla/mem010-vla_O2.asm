
/Users/jim/work/cppfort/micro-tests/results/memory/mem010-vla/mem010-vla_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z8test_vlai>:
100000498: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000049c: 910003fd    	mov	x29, sp
1000004a0: d10043ff    	sub	sp, sp, #0x10
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400908    	ldr	x8, [x8, #0x10]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: d37e7c09    	ubfiz	x9, x0, #2, #32
1000004b8: 91003d28    	add	x8, x9, #0xf
1000004bc: 927c7908    	and	x8, x8, #0x7fffffff0
1000004c0: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000004c4: f9400210    	ldr	x16, [x16]
1000004c8: d63f0200    	blr	x16
1000004cc: 910003e9    	mov	x9, sp
1000004d0: cb080128    	sub	x8, x9, x8
1000004d4: 9100011f    	mov	sp, x8
1000004d8: 7100041f    	cmp	w0, #0x1
1000004dc: 540000cb    	b.lt	0x1000004f4 <__Z8test_vlai+0x5c>
1000004e0: 2a0003e9    	mov	w9, w0
1000004e4: 7100101f    	cmp	w0, #0x4
1000004e8: 54000082    	b.hs	0x1000004f8 <__Z8test_vlai+0x60>
1000004ec: d280000a    	mov	x10, #0x0               ; =0
1000004f0: 14000029    	b	0x100000594 <__Z8test_vlai+0xfc>
1000004f4: 1400002e    	b	0x1000005ac <__Z8test_vlai+0x114>
1000004f8: 9000000b    	adrp	x11, 0x100000000 <___stack_chk_guard+0x100000000>
1000004fc: 7100401f    	cmp	w0, #0x10
100000500: 54000062    	b.hs	0x10000050c <__Z8test_vlai+0x74>
100000504: d280000a    	mov	x10, #0x0               ; =0
100000508: 14000015    	b	0x10000055c <__Z8test_vlai+0xc4>
10000050c: 927c692a    	and	x10, x9, #0x7ffffff0
100000510: 3dc17d60    	ldr	q0, [x11, #0x5f0]
100000514: 9100810c    	add	x12, x8, #0x20
100000518: 4f000481    	movi.4s	v1, #0x4
10000051c: 4f000502    	movi.4s	v2, #0x8
100000520: 4f000583    	movi.4s	v3, #0xc
100000524: 4f000604    	movi.4s	v4, #0x10
100000528: aa0a03ed    	mov	x13, x10
10000052c: 4ea18405    	add.4s	v5, v0, v1
100000530: 4ea28406    	add.4s	v6, v0, v2
100000534: 4ea38407    	add.4s	v7, v0, v3
100000538: ad3f1580    	stp	q0, q5, [x12, #-0x20]
10000053c: ac821d86    	stp	q6, q7, [x12], #0x40
100000540: 4ea48400    	add.4s	v0, v0, v4
100000544: f10041ad    	subs	x13, x13, #0x10
100000548: 54ffff21    	b.ne	0x10000052c <__Z8test_vlai+0x94>
10000054c: eb09015f    	cmp	x10, x9
100000550: 540002a0    	b.eq	0x1000005a4 <__Z8test_vlai+0x10c>
100000554: f27e053f    	tst	x9, #0xc
100000558: 540001e0    	b.eq	0x100000594 <__Z8test_vlai+0xfc>
10000055c: aa0a03ec    	mov	x12, x10
100000560: 4e040d80    	dup.4s	v0, w12
100000564: 927e712a    	and	x10, x9, #0x7ffffffc
100000568: 3dc17d61    	ldr	q1, [x11, #0x5f0]
10000056c: 4ea11c00    	orr.16b	v0, v0, v1
100000570: 8b0c090b    	add	x11, x8, x12, lsl #2
100000574: cb0a018c    	sub	x12, x12, x10
100000578: 4f000481    	movi.4s	v1, #0x4
10000057c: 3c810560    	str	q0, [x11], #0x10
100000580: 4ea18400    	add.4s	v0, v0, v1
100000584: b100118c    	adds	x12, x12, #0x4
100000588: 54ffffa1    	b.ne	0x10000057c <__Z8test_vlai+0xe4>
10000058c: eb09015f    	cmp	x10, x9
100000590: 540000a0    	b.eq	0x1000005a4 <__Z8test_vlai+0x10c>
100000594: b82a790a    	str	w10, [x8, x10, lsl #2]
100000598: 9100054a    	add	x10, x10, #0x1
10000059c: eb0a013f    	cmp	x9, x10
1000005a0: 54ffffa1    	b.ne	0x100000594 <__Z8test_vlai+0xfc>
1000005a4: 8b090908    	add	x8, x8, x9, lsl #2
1000005a8: b85fc100    	ldur	w0, [x8, #-0x4]
1000005ac: f85f83a8    	ldur	x8, [x29, #-0x8]
1000005b0: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
1000005b4: f9400929    	ldr	x9, [x9, #0x10]
1000005b8: f9400129    	ldr	x9, [x9]
1000005bc: eb08013f    	cmp	x9, x8
1000005c0: 54000081    	b.ne	0x1000005d0 <__Z8test_vlai+0x138>
1000005c4: 910003bf    	mov	sp, x29
1000005c8: a8c17bfd    	ldp	x29, x30, [sp], #0x10
1000005cc: d65f03c0    	ret
1000005d0: 94000003    	bl	0x1000005dc <___stack_chk_guard+0x1000005dc>

00000001000005d4 <_main>:
1000005d4: 52800080    	mov	w0, #0x4                ; =4
1000005d8: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000005dc <__stubs>:
1000005dc: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000005e0: f9400610    	ldr	x16, [x16, #0x8]
1000005e4: d61f0200    	br	x16

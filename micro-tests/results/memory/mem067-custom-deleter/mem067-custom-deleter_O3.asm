
/Users/jim/work/cppfort/micro-tests/results/memory/mem067-custom-deleter/mem067-custom-deleter_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000538 <__Z14custom_deleterPi>:
100000538: b4000040    	cbz	x0, 0x100000540 <__Z14custom_deleterPi+0x8>
10000053c: 1400006d    	b	0x1000006f0 <_strcmp+0x1000006f0>
100000540: d65f03c0    	ret

0000000100000544 <__Z19test_custom_deleterv>:
100000544: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000548: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000054c: 910043fd    	add	x29, sp, #0x10
100000550: 52800080    	mov	w0, #0x4                ; =4
100000554: 9400006a    	bl	0x1000006fc <_strcmp+0x1000006fc>
100000558: aa0003f3    	mov	x19, x0
10000055c: 52800548    	mov	w8, #0x2a               ; =42
100000560: b9000008    	str	w8, [x0]
100000564: 52800500    	mov	w0, #0x28               ; =40
100000568: 94000065    	bl	0x1000006fc <_strcmp+0x1000006fc>
10000056c: aa0003e8    	mov	x8, x0
100000570: f8008d1f    	str	xzr, [x8, #0x8]!
100000574: 90000029    	adrp	x9, 0x100004000 <_strcmp+0x100004000>
100000578: 91016129    	add	x9, x9, #0x58
10000057c: 91004129    	add	x9, x9, #0x10
100000580: f9000009    	str	x9, [x0]
100000584: a9014c1f    	stp	xzr, x19, [x0, #0x10]
100000588: 90000009    	adrp	x9, 0x100000000 <_strcmp+0x100000000>
10000058c: 9114e129    	add	x9, x9, #0x538
100000590: f9001009    	str	x9, [x0, #0x20]
100000594: 92800009    	mov	x9, #-0x1               ; =-1
100000598: f8e90108    	ldaddal	x9, x8, [x8]
10000059c: b40000a8    	cbz	x8, 0x1000005b0 <__Z19test_custom_deleterv+0x6c>
1000005a0: 52800540    	mov	w0, #0x2a               ; =42
1000005a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005a8: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005ac: d65f03c0    	ret
1000005b0: f9400008    	ldr	x8, [x0]
1000005b4: f9400908    	ldr	x8, [x8, #0x10]
1000005b8: aa0003f3    	mov	x19, x0
1000005bc: d63f0100    	blr	x8
1000005c0: aa1303e0    	mov	x0, x19
1000005c4: 94000042    	bl	0x1000006cc <_strcmp+0x1000006cc>
1000005c8: 52800540    	mov	w0, #0x2a               ; =42
1000005cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000005d0: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000005d4: d65f03c0    	ret
1000005d8: 9400004c    	bl	0x100000708 <_strcmp+0x100000708>
1000005dc: aa1303e0    	mov	x0, x19
1000005e0: 94000044    	bl	0x1000006f0 <_strcmp+0x1000006f0>
1000005e4: 9400004f    	bl	0x100000720 <_strcmp+0x100000720>
1000005e8: d4200020    	brk	#0x1
1000005ec: aa0003f3    	mov	x19, x0
1000005f0: 94000049    	bl	0x100000714 <_strcmp+0x100000714>
1000005f4: aa1303e0    	mov	x0, x19
1000005f8: 94000032    	bl	0x1000006c0 <_strcmp+0x1000006c0>
1000005fc: 94000002    	bl	0x100000604 <___clang_call_terminate>

0000000100000600 <_main>:
100000600: 17ffffd1    	b	0x100000544 <__Z19test_custom_deleterv>

0000000100000604 <___clang_call_terminate>:
100000604: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000608: 910003fd    	mov	x29, sp
10000060c: 9400003f    	bl	0x100000708 <_strcmp+0x100000708>
100000610: 94000035    	bl	0x1000006e4 <_strcmp+0x1000006e4>

0000000100000614 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEED1Ev>:
100000614: 14000031    	b	0x1000006d8 <_strcmp+0x1000006d8>

0000000100000618 <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEED0Ev>:
100000618: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
10000061c: 910003fd    	mov	x29, sp
100000620: 9400002e    	bl	0x1000006d8 <_strcmp+0x1000006d8>
100000624: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000628: 14000032    	b	0x1000006f0 <_strcmp+0x1000006f0>

000000010000062c <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE16__on_zero_sharedEv>:
10000062c: a9bf7bfd    	stp	x29, x30, [sp, #-0x10]!
100000630: 910003fd    	mov	x29, sp
100000634: a941a000    	ldp	x0, x8, [x0, #0x18]
100000638: d63f0100    	blr	x8
10000063c: a8c17bfd    	ldp	x29, x30, [sp], #0x10
100000640: d65f03c0    	ret
100000644: 97fffff0    	bl	0x100000604 <___clang_call_terminate>

0000000100000648 <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info>:
100000648: f9400428    	ldr	x8, [x1, #0x8]
10000064c: 90000009    	adrp	x9, 0x100000000 <_strcmp+0x100000000>
100000650: 911ebd29    	add	x9, x9, #0x7af
100000654: d2f0000a    	mov	x10, #-0x8000000000000000 ; =-9223372036854775808
100000658: 8b0a012a    	add	x10, x9, x10
10000065c: eb0a011f    	cmp	x8, x10
100000660: 54000061    	b.ne	0x10000066c <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info+0x24>
100000664: 91008000    	add	x0, x0, #0x20
100000668: d65f03c0    	ret
10000066c: ea0a011f    	tst	x8, x10
100000670: 5400006b    	b.lt	0x10000067c <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info+0x34>
100000674: d2800000    	mov	x0, #0x0                ; =0
100000678: d65f03c0    	ret
10000067c: a9be4ff4    	stp	x20, x19, [sp, #-0x20]!
100000680: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000684: 910043fd    	add	x29, sp, #0x10
100000688: d2f0000a    	mov	x10, #-0x8000000000000000 ; =-9223372036854775808
10000068c: 8b0a0129    	add	x9, x9, x10
100000690: aa0003f3    	mov	x19, x0
100000694: 9240f900    	and	x0, x8, #0x7fffffffffffffff
100000698: 9240f921    	and	x1, x9, #0x7fffffffffffffff
10000069c: 94000024    	bl	0x10000072c <_strcmp+0x10000072c>
1000006a0: aa0003e8    	mov	x8, x0
1000006a4: aa1303e0    	mov	x0, x19
1000006a8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000006ac: a8c24ff4    	ldp	x20, x19, [sp], #0x20
1000006b0: 34fffda8    	cbz	w8, 0x100000664 <__ZNKSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE13__get_deleterERKSt9type_info+0x1c>
1000006b4: d2800000    	mov	x0, #0x0                ; =0
1000006b8: d65f03c0    	ret

00000001000006bc <__ZNSt3__120__shared_ptr_pointerIPiPFvS1_ENS_9allocatorIiEEE21__on_zero_shared_weakEv>:
1000006bc: 1400000d    	b	0x1000006f0 <_strcmp+0x1000006f0>

Disassembly of section __TEXT,__stubs:

00000001000006c0 <__stubs>:
1000006c0: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
1000006c4: f9401e10    	ldr	x16, [x16, #0x38]
1000006c8: d61f0200    	br	x16
1000006cc: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
1000006d0: f9400210    	ldr	x16, [x16]
1000006d4: d61f0200    	br	x16
1000006d8: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
1000006dc: f9400610    	ldr	x16, [x16, #0x8]
1000006e0: d61f0200    	br	x16
1000006e4: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
1000006e8: f9400a10    	ldr	x16, [x16, #0x10]
1000006ec: d61f0200    	br	x16
1000006f0: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
1000006f4: f9402610    	ldr	x16, [x16, #0x48]
1000006f8: d61f0200    	br	x16
1000006fc: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000700: f9402a10    	ldr	x16, [x16, #0x50]
100000704: d61f0200    	br	x16
100000708: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
10000070c: f9400e10    	ldr	x16, [x16, #0x18]
100000710: d61f0200    	br	x16
100000714: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000718: f9401210    	ldr	x16, [x16, #0x20]
10000071c: d61f0200    	br	x16
100000720: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000724: f9401610    	ldr	x16, [x16, #0x28]
100000728: d61f0200    	br	x16
10000072c: 90000030    	adrp	x16, 0x100004000 <_strcmp+0x100004000>
100000730: f9402210    	ldr	x16, [x16, #0x40]
100000734: d61f0200    	br	x16

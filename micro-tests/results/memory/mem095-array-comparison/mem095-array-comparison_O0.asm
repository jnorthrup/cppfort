
/Users/jim/work/cppfort/micro-tests/results/memory/mem095-array-comparison/mem095-array-comparison_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z21test_array_comparisonv>:
100000498: d10143ff    	sub	sp, sp, #0x50
10000049c: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000004a0: 910103fd    	add	x29, sp, #0x40
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9116b108    	add	x8, x8, #0x5ac
1000004bc: f9400109    	ldr	x9, [x8]
1000004c0: f81e83a9    	stur	x9, [x29, #-0x18]
1000004c4: b9400908    	ldr	w8, [x8, #0x8]
1000004c8: b81f03a8    	stur	w8, [x29, #-0x10]
1000004cc: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004d0: 9116e108    	add	x8, x8, #0x5b8
1000004d4: f9400109    	ldr	x9, [x8]
1000004d8: f9000fe9    	str	x9, [sp, #0x18]
1000004dc: b9400908    	ldr	w8, [x8, #0x8]
1000004e0: b90023e8    	str	w8, [sp, #0x20]
1000004e4: b90013ff    	str	wzr, [sp, #0x10]
1000004e8: 14000001    	b	0x1000004ec <__Z21test_array_comparisonv+0x54>
1000004ec: b94013e8    	ldr	w8, [sp, #0x10]
1000004f0: 71000d08    	subs	w8, w8, #0x3
1000004f4: 5400024a    	b.ge	0x10000053c <__Z21test_array_comparisonv+0xa4>
1000004f8: 14000001    	b	0x1000004fc <__Z21test_array_comparisonv+0x64>
1000004fc: b98013e9    	ldrsw	x9, [sp, #0x10]
100000500: d10063a8    	sub	x8, x29, #0x18
100000504: b8697908    	ldr	w8, [x8, x9, lsl #2]
100000508: b98013ea    	ldrsw	x10, [sp, #0x10]
10000050c: 910063e9    	add	x9, sp, #0x18
100000510: b86a7929    	ldr	w9, [x9, x10, lsl #2]
100000514: 6b090108    	subs	w8, w8, w9
100000518: 54000080    	b.eq	0x100000528 <__Z21test_array_comparisonv+0x90>
10000051c: 14000001    	b	0x100000520 <__Z21test_array_comparisonv+0x88>
100000520: b90017ff    	str	wzr, [sp, #0x14]
100000524: 14000009    	b	0x100000548 <__Z21test_array_comparisonv+0xb0>
100000528: 14000001    	b	0x10000052c <__Z21test_array_comparisonv+0x94>
10000052c: b94013e8    	ldr	w8, [sp, #0x10]
100000530: 11000508    	add	w8, w8, #0x1
100000534: b90013e8    	str	w8, [sp, #0x10]
100000538: 17ffffed    	b	0x1000004ec <__Z21test_array_comparisonv+0x54>
10000053c: 52800028    	mov	w8, #0x1                ; =1
100000540: b90017e8    	str	w8, [sp, #0x14]
100000544: 14000001    	b	0x100000548 <__Z21test_array_comparisonv+0xb0>
100000548: b94017e8    	ldr	w8, [sp, #0x14]
10000054c: b9000fe8    	str	w8, [sp, #0xc]
100000550: f85f83a9    	ldur	x9, [x29, #-0x8]
100000554: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000558: f9400108    	ldr	x8, [x8]
10000055c: f9400108    	ldr	x8, [x8]
100000560: eb090108    	subs	x8, x8, x9
100000564: 54000060    	b.eq	0x100000570 <__Z21test_array_comparisonv+0xd8>
100000568: 14000001    	b	0x10000056c <__Z21test_array_comparisonv+0xd4>
10000056c: 9400000d    	bl	0x1000005a0 <___stack_chk_guard+0x1000005a0>
100000570: b9400fe0    	ldr	w0, [sp, #0xc]
100000574: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000578: 910143ff    	add	sp, sp, #0x50
10000057c: d65f03c0    	ret

0000000100000580 <_main>:
100000580: d10083ff    	sub	sp, sp, #0x20
100000584: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000588: 910043fd    	add	x29, sp, #0x10
10000058c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000590: 97ffffc2    	bl	0x100000498 <__Z21test_array_comparisonv>
100000594: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000598: 910083ff    	add	sp, sp, #0x20
10000059c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000005a0 <__stubs>:
1000005a0: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000005a4: f9400610    	ldr	x16, [x16, #0x8]
1000005a8: d61f0200    	br	x16

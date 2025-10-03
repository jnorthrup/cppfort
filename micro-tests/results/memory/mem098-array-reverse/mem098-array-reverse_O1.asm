
/Users/jim/work/cppfort/micro-tests/results/memory/mem098-array-reverse/mem098-array-reverse_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z18test_array_reversev>:
100000498: d100c3ff    	sub	sp, sp, #0x30
10000049c: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004a0: 910083fd    	add	x29, sp, #0x20
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400508    	ldr	x8, [x8, #0x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9116d108    	add	x8, x8, #0x5b4
1000004bc: 3dc00100    	ldr	q0, [x8]
1000004c0: 3d8003e0    	str	q0, [sp]
1000004c4: 528000a8    	mov	w8, #0x5                ; =5
1000004c8: b90013e8    	str	w8, [sp, #0x10]
1000004cc: 910003e8    	mov	x8, sp
1000004d0: 52800209    	mov	w9, #0x10               ; =16
1000004d4: 910003ea    	mov	x10, sp
1000004d8: b940014b    	ldr	w11, [x10]
1000004dc: b869690c    	ldr	w12, [x8, x9]
1000004e0: b800454c    	str	w12, [x10], #0x4
1000004e4: b829690b    	str	w11, [x8, x9]
1000004e8: d1001129    	sub	x9, x9, #0x4
1000004ec: f100313f    	cmp	x9, #0xc
1000004f0: 54ffff40    	b.eq	0x1000004d8 <__Z18test_array_reversev+0x40>
1000004f4: b94003e0    	ldr	w0, [sp]
1000004f8: f85f83a8    	ldur	x8, [x29, #-0x8]
1000004fc: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000500: f9400529    	ldr	x9, [x9, #0x8]
100000504: f9400129    	ldr	x9, [x9]
100000508: eb08013f    	cmp	x9, x8
10000050c: 54000081    	b.ne	0x10000051c <__Z18test_array_reversev+0x84>
100000510: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000514: 9100c3ff    	add	sp, sp, #0x30
100000518: d65f03c0    	ret
10000051c: 94000023    	bl	0x1000005a8 <___stack_chk_guard+0x1000005a8>

0000000100000520 <_main>:
100000520: d100c3ff    	sub	sp, sp, #0x30
100000524: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000528: 910083fd    	add	x29, sp, #0x20
10000052c: d2800008    	mov	x8, #0x0                ; =0
100000530: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000534: f9400529    	ldr	x9, [x9, #0x8]
100000538: f9400129    	ldr	x9, [x9]
10000053c: f81f83a9    	stur	x9, [x29, #-0x8]
100000540: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
100000544: 9116d129    	add	x9, x9, #0x5b4
100000548: 3dc00120    	ldr	q0, [x9]
10000054c: 3d8003e0    	str	q0, [sp]
100000550: 528000a9    	mov	w9, #0x5                ; =5
100000554: b90013e9    	str	w9, [sp, #0x10]
100000558: 910003e9    	mov	x9, sp
10000055c: 9100412a    	add	x10, x9, #0x10
100000560: b868692b    	ldr	w11, [x9, x8]
100000564: b940014c    	ldr	w12, [x10]
100000568: b828692c    	str	w12, [x9, x8]
10000056c: b81fc54b    	str	w11, [x10], #-0x4
100000570: 91001108    	add	x8, x8, #0x4
100000574: f100111f    	cmp	x8, #0x4
100000578: 54ffff40    	b.eq	0x100000560 <_main+0x40>
10000057c: b94003e0    	ldr	w0, [sp]
100000580: f85f83a8    	ldur	x8, [x29, #-0x8]
100000584: 90000029    	adrp	x9, 0x100004000 <___stack_chk_guard+0x100004000>
100000588: f9400529    	ldr	x9, [x9, #0x8]
10000058c: f9400129    	ldr	x9, [x9]
100000590: eb08013f    	cmp	x9, x8
100000594: 54000081    	b.ne	0x1000005a4 <_main+0x84>
100000598: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000059c: 9100c3ff    	add	sp, sp, #0x30
1000005a0: d65f03c0    	ret
1000005a4: 94000001    	bl	0x1000005a8 <___stack_chk_guard+0x1000005a8>

Disassembly of section __TEXT,__stubs:

00000001000005a8 <__stubs>:
1000005a8: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
1000005ac: f9400210    	ldr	x16, [x16]
1000005b0: d61f0200    	br	x16

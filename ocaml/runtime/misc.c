/**************************************************************************/
/*                                                                        */
/*                                 OCaml                                  */
/*                                                                        */
/*          Xavier Leroy and Damien Doligez, INRIA Rocquencourt           */
/*                                                                        */
/*   Copyright 1996 Institut National de Recherche en Informatique et     */
/*     en Automatique.                                                    */
/*                                                                        */
/*   All rights reserved.  This file is distributed under the terms of    */
/*   the GNU Lesser General Public License version 2.1, with the          */
/*   special exception on linking described in the file LICENSE.          */
/*                                                                        */
/**************************************************************************/

#define CAML_INTERNALS

#if _MSC_VER >= 1400 && _MSC_VER < 1700
/* Microsoft introduced a regression in Visual Studio 2005 (technically it's
   not present in the Windows Server 2003 SDK which has a pre-release version)
   and the abort function ceased to be declared __declspec(noreturn). This was
   fixed in Visual Studio 2012. Trick stdlib.h into not defining abort (this
   means exit and _exit are not defined either, but they aren't required). */
#define _CRT_TERMINATE_DEFINED
__declspec(noreturn) void __cdecl abort(void);
#endif

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "caml/config.h"
#include "caml/misc.h"
#include "caml/memory.h"
#include "caml/osdeps.h"
#include "caml/version.h"

caml_timing_hook caml_major_slice_begin_hook = NULL;
caml_timing_hook caml_major_slice_end_hook = NULL;
caml_timing_hook caml_minor_gc_begin_hook = NULL;
caml_timing_hook caml_minor_gc_end_hook = NULL;
caml_timing_hook caml_finalise_begin_hook = NULL;
caml_timing_hook caml_finalise_end_hook = NULL;

#ifdef DEBUG

void caml_failed_assert (char * expr, char_os * file_os, int line)
{
  char* file = caml_stat_strdup_of_os(file_os);
  fprintf (stderr, "file %s; line %d ### Assertion failed: %s\n",
           file, line, expr);
  fflush (stderr);
  caml_stat_free(file);
  abort();
}

void caml_set_fields (value v, uintnat start, uintnat filler)
{
  mlsize_t i;
  for (i = start; i < Wosize_val (v); i++){
    Field (v, i) = (value) filler;
  }
}

#endif /* DEBUG */

uintnat caml_verb_gc = 0;

void caml_gc_message (int level, char *msg, ...)
{
  if ((caml_verb_gc & level) != 0){
    va_list ap;
    va_start(ap, msg);
    if (caml_verb_gc & 0x1000) {
      caml_print_timestamp(stderr, caml_verb_gc & 0x2000);
    }
    vfprintf (stderr, msg, ap);
    va_end(ap);
    fflush (stderr);
  }
}

void (*caml_fatal_error_hook) (char *msg, va_list args) = NULL;

CAMLexport void caml_fatal_error (char *msg, ...)
{
  va_list ap;
  va_start(ap, msg);
  if(caml_fatal_error_hook != NULL) {
    caml_fatal_error_hook(msg, ap);
  } else {
    fprintf (stderr, "Fatal error: ");
    vfprintf (stderr, msg, ap);
    fprintf (stderr, "\n");
  }
  va_end(ap);
  abort();
}

void caml_fatal_out_of_memory(void)
{
  caml_fatal_error("Out of memory");
}

void caml_ext_table_init(struct ext_table * tbl, int init_capa)
{
  tbl->size = 0;
  tbl->capacity = init_capa;
  tbl->contents = caml_stat_alloc(sizeof(void *) * init_capa);
}

int caml_ext_table_add(struct ext_table * tbl, caml_stat_block data)
{
  int res;
  if (tbl->size >= tbl->capacity) {
    tbl->capacity *= 2;
    tbl->contents =
      caml_stat_resize(tbl->contents, sizeof(void *) * tbl->capacity);
  }
  res = tbl->size;
  tbl->contents[res] = data;
  tbl->size++;
  return res;
}

void caml_ext_table_remove(struct ext_table * tbl, caml_stat_block data)
{
  int i;
  for (i = 0; i < tbl->size; i++) {
    if (tbl->contents[i] == data) {
      caml_stat_free(tbl->contents[i]);
      memmove(&tbl->contents[i], &tbl->contents[i + 1],
              (tbl->size - i - 1) * sizeof(void *));
      tbl->size--;
    }
  }
}

void caml_ext_table_clear(struct ext_table * tbl, int free_entries)
{
  int i;
  if (free_entries) {
    for (i = 0; i < tbl->size; i++) caml_stat_free(tbl->contents[i]);
  }
  tbl->size = 0;
}

void caml_ext_table_free(struct ext_table * tbl, int free_entries)
{
  caml_ext_table_clear(tbl, free_entries);
  caml_stat_free(tbl->contents);
}

/* Integer arithmetic with overflow detection */

#if ! (__GNUC__ >= 5 || Caml_has_builtin(__builtin_mul_overflow))
CAMLexport int caml_umul_overflow(uintnat a, uintnat b, uintnat * res)
{
#define HALF_SIZE (sizeof(uintnat) * 4)
#define HALF_MASK (((uintnat)1 << HALF_SIZE) - 1)
#define LOW_HALF(x) ((x) & HALF_MASK)
#define HIGH_HALF(x) ((x) >> HALF_SIZE)
  /* Cut in half words */
  uintnat al = LOW_HALF(a);
  uintnat ah = HIGH_HALF(a);
  uintnat bl = LOW_HALF(b);
  uintnat bh = HIGH_HALF(b);
  /* Exact product is:
              al * bl
           +  ah * bl  << HALF_SIZE
           +  al * bh  << HALF_SIZE
           +  ah * bh  << 2*HALF_SIZE
     Overflow occurs if:
        ah * bh is not 0, i.e. ah != 0 and bh != 0
     OR ah * bl has high half != 0
     OR al * bh has high half != 0
     OR the sum al * bl + LOW_HALF(ah * bl) << HALF_SIZE
                        + LOW_HALF(al * bh) << HALF_SIZE overflows.
     This sum is equal to p = (a * b) modulo word size. */
  uintnat p = a * b;
  uintnat p1 = al * bh;
  uintnat p2 = ah * bl;
  *res = p;
  if (ah == 0 && bh == 0) return 0;
  if (ah != 0 && bh != 0) return 1;
  if (HIGH_HALF(p1) != 0 || HIGH_HALF(p2) != 0) return 1;
  p1 <<= HALF_SIZE;
  p2 <<= HALF_SIZE;
  p1 += p2;
  if (p < p1 || p1 < p2) return 1; /* overflow in sums */
  return 0;
#undef HALF_SIZE
#undef HALF_MASK
#undef LOW_HALF
#undef HIGH_HALF
}
#endif

/* Runtime warnings */

uintnat caml_runtime_warnings = 0;
static int caml_runtime_warnings_first = 1;

int caml_runtime_warnings_active(void)
{
  if (!caml_runtime_warnings) return 0;
  if (caml_runtime_warnings_first) {
    fprintf(stderr, "[ocaml] (use Sys.enable_runtime_warnings to control "
                    "these warnings)\n");
    caml_runtime_warnings_first = 0;
  }
  return 1;
}

/* Flambda 2 invalid term markers */

CAMLnoreturn_start
void caml_flambda2_invalid (value message)
CAMLnoreturn_end;

void caml_flambda2_invalid (value message)
{
  fprintf (stderr, "[ocaml] [flambda2] Invalid code:\n%s\n\n",
    String_val(message));
  fprintf (stderr, "This might have arisen from a wrong use of [Obj.magic].\n");
  fprintf (stderr, "Consider using [Sys.opaque_identity].\n");
  abort ();
}

/* Functions used by caml_curry_generic */

/* See diagram below */
#define VARARGS_BUFFER_HEADER_SIZE 4

/* Helper function for fishing parameters out of closures and assembling
   them into memory blocks for a variadic call in caml_curry_generic.

   This function is called recursively, starting with the newest closure
   created by caml_curry_generic and working its way back to the oldest
   closure.  It returns the actual closure for the function ultimately
   being called. */
static value extricate_parameters (value closure, uintnat* buffer,
  /* Total number of unarized params to be passed in registers: */
  uintnat num_int_in_regs, uintnat num_float_in_regs,
  /* 1 if the closure argument will be passed in a register, otherwise 0: */
  int closure_arg_passed_in_reg,
  /* Number of unarized params written to [buffer]; note that
     [num_int_written] and [num_float_written] include all such arguments,
     not just the ones in registers: */
  uintnat* num_int_written, uintnat* num_float_written,
  uintnat* num_stack_written,
  /* Function pointer of the actual closure for the function being called.
     The closure itself is the return value. */
  uintnat* func_ptr)
{
  uintnat startenv;
  value parent_closure;
  value actual_closure;
  uintnat num_complex_params_in_earlier_closures;
  uintnat* layout;
  uintnat* layout_this_complex_param;
  uintnat* clos_field_non_scannable;
  value* clos_field_scannable;

  startenv = Start_env_closinfo(Closinfo_val(closure));
  /* caml_generic_curry closures are always of arity 1 */
  CAMLassert(Arity_closinfo(Closinfo_val(closure)) == 1);
  CAMLassert(Wosize_val(closure) >= 2);

  /* For example, when caml_curry_generic creates a closure with the num-seen
     value equal to 1, that means such closure corresponds to the first
     complex parameter.  (Recall that caml_curry_generic applications are
     always done one complex parameter at a time.) */
  num_complex_params_in_earlier_closures =
    (uintnat) Field(closure, startenv - 2);

  if (num_complex_params_in_earlier_closures > 0) {
    /* This is the recursive case: move to the previous (earlier) closure. */

    parent_closure = Field(closure, startenv);
    CAMLassert(Is_block(parent_closure));
    /* Note that [parent_closure] should never have tag [Infix_tag]. */
    CAMLassert(Tag_val(parent_closure) == Closure_tag);

    actual_closure = extricate_parameters(parent_closure, buffer,
      num_int_in_regs, num_float_in_regs,
      closure_arg_passed_in_reg,
      num_int_written, num_float_written,
      num_stack_written, func_ptr);
  }
  else {
    /* At this point we've either reached the base case of the recursion,
       i.e. the oldest closure made by caml_curry_generic. */

    /* Extract the actual closure for the function ultimately being called. */
    actual_closure = Field(closure, startenv);

    CAMLassert(Is_block(actual_closure));
    CAMLassert(Tag_val(actual_closure) == Closure_tag
      || Tag_val(actual_closure) == Infix_tag);
    CAMLassert(Arity_closinfo(Closinfo_val(actual_closure)) > 1);
    CAMLassert(Wosize_val(actual_closure) >= 3);

    /* Extract the full application code pointer. */
    *func_ptr = (uintnat) Field(actual_closure, 2);
  }

  layout = (uintnat*) Field(closure, startenv - 1);

  layout_this_complex_param =
    (uintnat*) (((unsigned char*) layout)
      + layout[num_complex_params_in_earlier_closures - 1]);

  /* Traverse the zero-terminated array for the current complex parameter
     and copy the corresponding stored unarized arguments into the buffer. */
  clos_field_non_scannable = (uintnat*) &Field(closure, 2);
  clos_field_scannable = &Field(closure, startenv + 1 /* skip closure link */);

  /* Recall: only one complex parameter's arguments are in any one of the
     closures in the list. */
  buffer += VARARGS_BUFFER_HEADER_SIZE;
  while (*layout_this_complex_param != NULL) {
    uintnat int_index = *num_int_written;
    uintnat float_base = num_int_in_regs + closure_arg_passed_in_reg;
    uintnat float_index = float_base + *num_float_written;
    uintnat stack_index = float_base + num_float_in_regs + *num_stack_written;

    int room_in_int_regs =
      *num_int_written < (num_int_in_regs - closure_arg_passed_in_reg);

    int room_in_float_regs = *num_float_written < num_float_in_regs;

    switch (*layout_this_complex_param++) {
      case 1: { /* scannable */
        value v = *clos_field_scannable++;
        if (room_in_int_regs) {
          buffer[int_index] = (uintnat) v;
        } else {
          buffer[stack_index] = (uintnat) v;
          *num_stack_written += 1;
        }
        *num_int_written += 1;
        break;
      }

      case 2: { /* non-scannable int */
        uintnat v = *clos_field_non_scannable++;
        if (room_in_int_regs) {
          buffer[int_index] = v;
        } else {
          buffer[stack_index] = v;
          *num_stack_written += 1;
        }
        *num_int_written += 1;
        break;
      }

      case 3: { /* float */
        uintnat f = *clos_field_non_scannable++;
        if (room_in_float_regs) {
          buffer[float_index] = f;
        }
        else {
          buffer[stack_index] = f;
          *num_stack_written += 1;
        }
        *num_float_written += 1;
        break;
      }

      default:
        CAMLassert(0);
        abort();
    }
  }

  return actual_closure;
}

/* This function is called by the generated code for caml_curry_generic.
   It accepts a linked list of closures which contain all of the arguments
   making up a full function application.  The head of the linked list
   contains the arguments for the most recent partial application.

   It returns a buffer containing the function pointer and all of the
   unarized arguments for the function being called.  The caller is
   responsible for calling [free] on the buffer.

   The buffer is laid out as follows:

   -----------------------------------------------------------------------
   function pointer to be called
   num of unarized int register arguments
     (potentially including the closure argument)
   num of unarized float register arguments
   num of unarized stack and/or domainstate arguments (int or float)
     (potentially including the closure argument)
   -----------------------------------------------------------------------
   unarized int arguments (potentially including the closure argument)
   ...
   -----------------------------------------------------------------------
   unarized float arguments
   ...
   -----------------------------------------------------------------------
   unarized stack/domainstate args (potentially including the closure arg)
   (these args may be ints or floats freely intermixed)
   ...
   -----------------------------------------------------------------------

   The closure arg is always the last argument in the int area, or the
   last stack/domainstate argument.

   The caller is responsible for calling [free] on the buffer.
*/
uintnat* caml_curry_generic_helper (value callee_closure, value v_num_int_regs,
  value v_num_float_regs)
{
  uintnat startenv;
  uintnat* layout;
  uintnat num_int_in_regs, num_float_in_regs, num_stack;
  uintnat num_int_written, num_float_written, num_stack_written;
  uintnat complex_param_index;
  uintnat* buffer;
  uintnat func_ptr;
  value actual_closure;
  int closure_arg_passed_in_reg;
  uintnat total_complex_params;
  uintnat num_available_int_regs = Long_val(v_num_int_regs);
  uintnat num_available_float_regs = Long_val(v_num_float_regs);

  startenv = Start_env_closinfo(Closinfo_val(callee_closure));
  layout = (uintnat*) Field(callee_closure, startenv - 1);
  total_complex_params = (uintnat) Field(callee_closure, startenv - 2);

  /* These counters do not include the actual closure argument */
  num_int_in_regs = 0;
  num_float_in_regs = 0;
  num_stack = 0; /* stack and/or domainstate; ints and floats intermixed */

  /* First count the total number of int reg, float reg and stack/domainstate
     slots required to accommodate all of the arguments. */
  for (complex_param_index = 0; complex_param_index < total_complex_params;
       complex_param_index++) {
    uintnat* layout_this_complex_param =
      (uintnat*) (((unsigned char*) layout) + layout[complex_param_index]);

    while (*layout_this_complex_param != NULL) {
      switch (*layout_this_complex_param++) {
        case 1: /* scannable */
        case 2: /* non-scannable int */
          if (num_int_in_regs < num_available_int_regs) num_int_in_regs++;
          else num_stack++;
          break;

        case 3: /* float */
          if (num_float_in_regs < num_available_float_regs) num_float_in_regs++;
          else num_stack++;
          break;

        default:
          CAMLassert(0);
          abort();
      }
    }
  }

  /* The actual closure argument goes in the last int register, but that
     might be in the stack/domainstate area. */
  closure_arg_passed_in_reg =
    (num_int_in_regs < num_available_int_regs) ? 1 : 0;

  buffer = (uintnat*) malloc(sizeof(uintnat)
    * (num_int_in_regs + num_float_in_regs + num_stack
       + VARARGS_BUFFER_HEADER_SIZE + 1 /* closure arg */));
  if (buffer == NULL) {
    caml_fatal_out_of_memory ();
  }

  num_int_written = 0;
  num_float_written = 0;
  num_stack_written = 0;

  /* Go back to the oldest closure in the linked list of closures
     constructed by caml_curry_generic, then write all of the unarized
     parameters into [buffer], advancing through the linked list towards
     the newest closure.

     caml_curry_generic uses the first field of the scannable environment
     for the linked list of closures (corresponding to the sequence of
     partial applications).
  */
  actual_closure = extricate_parameters(Field(callee_closure, startenv),
    buffer, num_int_in_regs, num_float_in_regs, closure_arg_passed_in_reg,
    &num_int_written, &num_float_written,
    &num_stack_written, &func_ptr);

  buffer[0] = func_ptr;
  buffer[2] = num_float_in_regs;

  if (closure_arg_passed_in_reg) {
    buffer[VARARGS_BUFFER_HEADER_SIZE + num_int_in_regs]
      = (uintnat) actual_closure;
    buffer[1] = num_int_in_regs + 1;
    buffer[3] = num_stack;
  }
  else {
    buffer[VARARGS_BUFFER_HEADER_SIZE + num_int_in_regs + num_float_in_regs
        + num_stack - 1]
      = (uintnat) actual_closure;
    buffer[1] = num_int_in_regs;
    buffer[3] = num_stack + 1;
  }

  return buffer;
}

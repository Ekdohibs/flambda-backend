(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Binding times and name modes: when a name of a typing environment was
   bound, and where it may occur.

   This file is to be reviewed, line by line against
   middle_end/flambda2/types/env/binding_time.ml and
   middle_end/flambda2/nominal/name_mode.ml, as FlambdaModel.v is against
   flambda.mli. It holds definitions only, and it extracts. The typing
   environment gives each name a binding time and a name mode, and the
   aliases of the environment order names by them (FlambdaEnvAliases.v).

   The modules have the compiler's names, so that
   [Binding_time.With_name_mode.scoped_name_mode] is that function.

   What the model has, and how it differs from the compiler:
   - [Name_mode.t], its three constructors in the compiler's order, with
     [compare_partial_order], which answers an [int option] in the
     compiler and an [option Z] here. In the compiler too it answers
     [Some] for every pair: the order is total, [Normal] above
     [Phantom] above [In_types]. [equal] is by cases, where the compiler
     defines it from its total order, which has the same table.
     [join_in_terms], [Or_absent] and the containers are left out.
   - [Binding_time.t] is an integer, as in the compiler, and here it is
     unbounded: [max_binding_time] is left out, and with it the fatal
     error of [succ] when binding times run out. The other fatal error
     of [succ], on a binding time below [earliest_var], is left out
     too: [succ] is the successor on every integer.
   - [Binding_time.With_name_mode.t] is a record of a binding time and
     a name mode. The compiler packs the two into one integer, the name
     mode in the two low bits; [create], [binding_time] and [name_mode]
     are its packing and unpacking, here the constructor and the
     projections. [equal] compares the two components. *)

From Stdlib Require Import ZArith Bool.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Name modes                                                               *)
(* ======================================================================== *)

Module Name_mode.

  (* [Name_mode.t]: a normal name has a value at run time; a phantom name
     is for debugging information and an in-types name exists only in
     types, and neither has a value at run time. *)
  Inductive t : Type :=
    | Normal
    | Phantom
    | In_types.

  Definition normal : t := Normal.

  Definition in_types : t := In_types.

  Definition phantom : t := Phantom.

  Definition is_normal (t : t) : bool :=
    match t with
    | Normal => true
    | In_types | Phantom => false
    end.

  Definition is_phantom (t : t) : bool :=
    match t with
    | Phantom => true
    | In_types | Normal => false
    end.

  Definition is_in_types (t : t) : bool :=
    match t with
    | In_types => true
    | Normal | Phantom => false
    end.

  Definition can_be_in_terms (t : t) : bool :=
    match t with
    | Normal | Phantom => true
    | In_types => false
    end.

  (* [Name_mode.compare_partial_order]: negative, zero or positive as the
     first is below, the same as, or above the second. *)
  Definition compare_partial_order (t1 t2 : t) : option Z :=
    match t1, t2 with
    | Normal, Normal | Phantom, Phantom | In_types, In_types => Some 0
    | Normal, (Phantom | In_types) => Some 1
    | (Phantom | In_types), Normal => Some (-1)
    | Phantom, In_types => Some 1
    | In_types, Phantom => Some (-1)
    end.

  (* [Name_mode.equal]. *)
  Definition equal (t1 t2 : t) : bool :=
    match t1, t2 with
    | Normal, Normal | Phantom, Phantom | In_types, In_types => true
    | _, _ => false
    end.

End Name_mode.

(* ======================================================================== *)
(* Binding times                                                            *)
(* ======================================================================== *)

Module Binding_time.

  (* [Binding_time.t]. *)
  Definition t : Type := Z.

  Definition strictly_earlier (t : t) (than : Binding_time.t) : bool :=
    t <? than.

  (* The binding time of every constant, of every symbol, and of every
     variable of another compilation unit; and the first binding time of
     a variable of this one. *)
  Definition consts : t := 0.

  Definition symbols : t := 1.

  Definition imported_variables : t := 2.

  Definition earliest_var : t := 3.

  Definition succ (t : t) : Binding_time.t := t + 1.

  Definition equal (t1 t2 : t) : bool := t1 =? t2.

  Module With_name_mode.

    (* [Binding_time.With_name_mode.t], with [create], [binding_time] and
       [name_mode]. *)
    Record t : Type := create {
      binding_time : Binding_time.t;
      name_mode : Name_mode.t
    }.

    Definition symbols : t := create Binding_time.symbols Name_mode.normal.

    Definition consts : t := create Binding_time.consts Name_mode.normal.

    Definition imported_variables : t :=
      create Binding_time.imported_variables Name_mode.in_types.

    (* The name mode of a name where only the variables bound at
       [min_binding_time] or later are in scope: a variable bound before
       then is out of scope, and may occur only in types. Constants,
       symbols and the variables of other compilation units are never out
       of scope. *)
    Definition scoped_name_mode (t : t) (min_binding_time : Binding_time.t)
        : Name_mode.t :=
      if (binding_time t <? Binding_time.earliest_var)
         || (min_binding_time <=? binding_time t)
      then name_mode t
      else Name_mode.in_types.

    Definition equal (t1 t2 : t) : bool :=
      Binding_time.equal (binding_time t1) (binding_time t2)
      && Name_mode.equal (name_mode t1) (name_mode t2).

  End With_name_mode.

End Binding_time.

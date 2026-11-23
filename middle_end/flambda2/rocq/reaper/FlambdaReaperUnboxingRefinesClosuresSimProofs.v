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

(* The relation of the proofs of FlambdaReaperUnboxingRefinesClosuresSpec.v
   between the unit rebuilt with the change of calling conventions alone
   (the first run) and the unit rebuilt with unboxing and changes of
   representation of sets of closures (the second run), and that it is a
   simulation ([rr_refines]).

   The names are of three classes. A plain name has one value in both
   runs. An unboxed name ([M]) is bound in the first run to a value whose
   fields, at their paths, the variables of its fields hold in the second
   (FlambdaReaperUnboxingRefinesProofs.v: [fields_hold], [env_rel]). A
   changed name ([CH]: a closure of a set whose representation changes)
   is bound in the first run to a closure at the slot [cr_current] of the
   representation, and in the second to the closure of the set made with
   the new slots ([crel]): at the new function slot, with the function
   declarations renamed, and holding at the new value slots of each field
   of the representation the value of the old slot, or, for a field
   unboxed in turn, the values at the paths of its fields. A changed name
   is in [M] with no field, so that it is not plain and [env_rel] asks
   nothing of it.

   The two terms are the same but where the second unboxes, as in
   FlambdaReaperUnboxingRefinesProofs.v ([rr_alloc], [rr_alias],
   [rr_load], [rr_read]), at the [Let] of a set whose representation
   changes ([rr_set]: the new slots, [new_slots]), and at a read of a
   value slot of a changed closure, which the second makes at the new
   slot ([rr_proj]), or at the new slots of the fields of an unboxed
   value ([rr_proj_unboxed]: [proj_lets]). Calls are the same in both
   ([rr_call]): direct, full, with plain arguments, and with a callee of
   the class of the [my_closure] of the code ([CS]: [callee_ok]), so that
   the code of both runs, related, runs with its [my_closure] related.
   Arguments of continuations are plain, so that the values of an exit
   are the same in both runs.

   The relation has dead continuations [D]: no case applies them or calls
   with them, and their frames need not be related ([conts_rel]). Code
   that never returns ([ci_never]) has its body related with its return
   continuation dead ([code_dead]). A call of it may return, in the
   second run, to a handler [Invalid] bound before the call, and in the
   first to its continuation ([rr_call_never_r]) or to a wrapper
   ([rr_call_never_wraps]): the frames of the return continuation of the
   code are then any. A handler that only the first run binds is dead
   ([rr_let_cont_never]). The free continuations of the second term are
   those of the first or dead, and it binds statics only where the first
   does ([rr_shape]); the dead continuations not free in it may change
   ([rr_dead_ext]).

   Each step of the first run is answered by as many of the second as it
   makes [Let]s, none where it makes none, and a call that never returns
   by the [Let_cont] of the second and its call, so that the relation is
   a shrinking simulation (FlambdaReaperRebuildProofs.shrinking_refines).
   *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaMapSpec FlambdaMapProofs
  FlambdaValueSpec FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaReaperGraphSpec
  FlambdaReaperRebuild FlambdaReaperRebuildProofs FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs.
Import ListNotations.

(* ======================================================================== *)
(* The relation                                                             *)
(* ======================================================================== *)

(* What a call of code needs of its callee: the class of the
   [my_closure] of the code (plain, or changed with that
   representation), the number of groups of its parameters, and whether
   the code never returns. *)
Record code_info : Type := Code_info {
  ci_my_closure : option closure_repr;
  ci_nparams : nat;
  ci_never : bool
}.

(* The continuations that the body of code does not apply: its return
   continuation, if it never returns. *)
Definition code_dead (ci : code_info) (c : code) : list continuation :=
  if ci_never ci then [code_return_continuation c] else [].

(* A read of a value slot, and from what. *)
Definition slot_read (d : named) : option (value_slot * simple) :=
  match d with
  | Project_value_slot w s | Project_value_slot_from _ w s => Some (w, s)
  | _ => None
  end.

(* The [Let]s that read the new slots [pairs] of a changed closure [y] at
   the function slot [cs], the first the innermost
   ([load_field_from_value_which_is_being_unboxed]). *)
Definition proj_lets (cs : function_slot) (y : simple)
    (pairs : list (variable * variable * flambda_kind)) (body : expr)
    : expr :=
  List.fold_left
    (fun body '(x, slot, kind) =>
       Let x kind (Project_value_slot_from cs slot y) body)
    pairs body.

Section Related.

Variable M : list (node * unboxed_fields).
Variable CH : variable -> option closure_repr.
Variable CS : code_id -> option code_info.

(* The new value slots of the field [fu] of a representation, for a set
   whose old value slots are [vs]: the new slot of a field not unboxed
   holds what the old one did; those of a field unboxed in turn hold the
   variables of the fields of the unboxed name that the old one held. *)
Definition new_slot (vs : list (value_slot * simple))
    (fu : field * unboxed_field) : list (value_slot * simple) :=
  match fu with
  | (Field_value_slot w, u) =>
    match lookup w vs with
    | Some a =>
      match u with
      | Not_unboxed s => [(s, a)]
      | Unboxed _ =>
        match a with
        | Var b =>
          match fields_of M b with
          | Some fb =>
            match unboxed_field_pairs Value u (Unboxed fb) with
            | Some pairs => map (fun '(s, x, _) => (s, Var x)) pairs
            | None => []
            end
          | None => []
          end
        | Symbol _ | Const _ => []
        end
      end
    | None => []
    end
  | _ => []
  end.

Definition new_slots (r : closure_repr) (vs : list (value_slot * simple))
    : list (value_slot * simple) :=
  flat_map (new_slot vs) (cr_fields r).

(* A field of a representation, for a set whose value slots are [vs]: a
   value slot of the set, holding a plain simple or, if the field is
   unboxed in turn, a name unboxed alike. *)
Definition set_field_ok (vs : list (value_slot * simple))
    (fu : field * unboxed_field) : Prop :=
  match fu with
  | (Field_value_slot w, u) =>
    ∃ a : simple,
      lookup w vs = Some a
      ∧ match u with
        | Not_unboxed _ => plain_simple M a = true
        | Unboxed _ =>
          ∃ (b : variable) (fb : unboxed_fields)
            (pairs : list (variable * variable * flambda_kind)),
            a = Var b ∧ CH b = None ∧ fields_of M b = Some fb
            ∧ unboxed_field_pairs Value u (Unboxed fb) = Some pairs
        end
  | _ => False
  end.

Definition callee_ok (ci : code_info) (s : simple) : Prop :=
  match ci_my_closure ci with
  | None => plain_simple M s = true
  | Some r => ∃ f : variable, s = Var f ∧ CH f = Some r
  end.

Definition my_closure_ok (ci : code_info) (z : variable) : Prop :=
  match ci_my_closure ci with
  | None => plain_variable M z = true
  | Some r => CH z = Some r
  end.

(* What the code of the two runs shares but for its body. *)
Definition code_frame (id : code_id) (c c' : code) (ci : code_info) : Prop :=
  CS id = Some ci
  ∧ code_params c' = code_params c
  ∧ code_my_closure c' = code_my_closure c
  ∧ code_return_continuation c' = code_return_continuation c
  ∧ code_exn_continuation c' = code_exn_continuation c
  ∧ code_params_arity c' = code_params_arity c
  ∧ code_result_arity c' = code_result_arity c
  ∧ code_is_tupled c = false ∧ code_is_tupled c' = false
  ∧ length (code_params_arity c) = ci_nparams ci
  ∧ forallb (plain_variable M) (code_params c) = true
  ∧ my_closure_ok ci (code_my_closure c)
  ∧ subkinds_say_nothing (code_params_subkinds c') = true.

(* A call of the code [id] whose class is [ci]: as [rr_call]. *)
Definition call_ok_c (ap : apply_expr) (id : code_id) (ci : code_info)
    : Prop :=
  ap_call_kind ap = Direct id ∧ CS id = Some ci ∧ callee_ok ci (ap_callee ap)
  ∧ forallb (plain_simple M) (ap_args ap) = true
  ∧ ap_exn_extra_args ap = [] ∧ length (ap_args_arity ap) = ci_nparams ci.

(* The terms related, the continuations [D] dead: no case applies them, or
   calls with them, and their frames are not related ([conts_rel]). *)
Inductive rr : list continuation -> expr -> expr -> Prop :=
  | rr_let (D : list continuation)
      (x : variable) (kind : flambda_kind) (d : named)
      (body body' : expr) :
    plain_variable M x = true
    -> FlambdaReaperUnboxingRefinesProofs.named_plain M d = true ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    rr D body body' -> rr D (Let x kind d body) (Let x kind d body')
  | rr_plain_set (D : list continuation)
      (x : variable) (kind : flambda_kind) (slot : function_slot)
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * simple)) (body body' : expr) :
    plain_variable M x = true ->
    forallb (fun ws => plain_simple M (snd ws)) vs = true ->
    rr D body body' ->
    rr D (Let x kind (Closure_of_set slot decls vs) body)
      (Let x kind (Closure_of_set slot decls vs) body')
  | rr_alloc (D : list continuation)
      (x : variable) (d : named) (fields : unboxed_fields)
      (ls : list (list (variable * flambda_kind * simple)))
      (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    Forall2 (alloc_entry M x d) fields ls ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    rr D body body' -> rr D (Let x Value d body) (lets (concat ls) body')
  | rr_alias (D : list continuation)
      (x y : variable) (fields fields_y : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)) (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    fields_of M y = Some fields_y -> x <> y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs ->
    rr D body body' ->
    rr D (Let x Value (Simple (Var y)) body) (lets (pairs_lets pairs) body')
  | rr_load (D : list continuation)
      (x : variable) (d : named) (f : field) (s : simple) (y : variable)
      (fields fields_y nested : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)) (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields_y -> x <> y ->
    (f, Unboxed nested) ∈ fields_y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed nested)
    = Some pairs ->
    rr D body body' -> rr D (Let x Value d body) (lets (pairs_lets pairs) body')
  | rr_read (D : list continuation)
      (x : variable) (kind : flambda_kind) (d : named) (f : field)
      (s : simple) (y v : variable) (fields : unboxed_fields)
      (body body' : expr) :
    plain_variable M x = true -> read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields -> (f, Not_unboxed v) ∈ fields ->
    rr D body body' ->
    rr D (Let x kind d body) (Let x kind (Simple (Var v)) body')
  | rr_set (D : list continuation)
      (x : variable) (r : closure_repr)
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * simple)) (body body' : expr) :
    CH x = Some r ->
    lookup (current_slot r) (rename_function_decls r decls)
    = lookup (cr_current r) decls ->
    Forall (set_field_ok vs) (cr_fields r) ->
    NoDup (map fst (new_slots r vs)) ->
    rr D body body' ->
    rr D (Let x Value (Closure_of_set (cr_current r) decls vs) body)
      (Let x Value
         (Closure_of_set (current_slot r) (rename_function_decls r decls)
            (new_slots r vs))
         body')
  | rr_proj (D : list continuation)
      (x : variable) (kind : flambda_kind) (d : named) (w : value_slot)
      (s : simple) (y : variable) (r : closure_repr) (sl : value_slot)
      (body body' : expr) :
    plain_variable M x = true -> slot_read d = Some (w, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    CH y = Some r -> (Field_value_slot w, Not_unboxed sl) ∈ cr_fields r ->
    rr D body body' ->
    rr D (Let x kind d body)
      (Let x kind (Project_value_slot_from (current_slot r) sl (Var y))
         body')
  | rr_proj_unboxed (D : list continuation)
      (x : variable) (d : named) (w : value_slot) (s : simple)
      (y : variable) (r : closure_repr) (fields nested : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)) (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    slot_read d = Some (w, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    CH y = Some r -> (Field_value_slot w, Unboxed nested) ∈ cr_fields r ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed nested)
    = Some pairs ->
    rr D body body' ->
    rr D (Let x Value d body) (proj_lets (current_slot r) (Var y) pairs body')
  | rr_static (D : list continuation) (bs bs' : list static_binding)
      (body body' : expr) :
    rr_bindings bs bs' -> rr D body body' ->
    rr D (Let_static bs body) (Let_static bs' body')
  | rr_let_cont (D : list continuation) (k : continuation)
      (ps : list variable) (md : handler_metadata) (h h' body body' : expr) :
    forallb (plain_variable M) ps = true -> rr D h h' ->
    rr (without [k] D) body body' ->
    rr D (Let_cont_non_recursive k (Handler ps md h) body)
      (Let_cont_non_recursive k (Handler ps md h') body')
  | rr_let_cont_rec (D : list continuation)
      (hs hs' : list (continuation * continuation_handler))
      (body body' : expr) :
    rr_handlers (without (map fst hs) D) hs hs' ->
    rr (without (map fst hs) D) body body' ->
    rr D (Let_cont_recursive hs body) (Let_cont_recursive hs' body')
  | rr_apply_cont (D : list continuation) (ac : apply_cont) :
    forallb (plain_simple M) (ac_args ac) = true -> ac_continuation ac ∉ D ->
    rr D (Apply_cont ac) (Apply_cont ac)
  | rr_switch (D : list continuation) (scrutinee : simple)
      (arms : list (Z * apply_cont)) :
    plain_simple M scrutinee = true ->
    forallb (fun arm => forallb (plain_simple M) (ac_args (snd arm))) arms
    = true ->
    (∀ arm : Z * apply_cont, arm ∈ arms -> ac_continuation (snd arm) ∉ D) ->
    rr D (Switch scrutinee arms) (Switch scrutinee arms)
  | rr_invalid (D : list continuation) : rr D Invalid Invalid
  | rr_call (D : list continuation) (ap : apply_expr) (id : code_id)
      (ci : code_info) :
    ap_call_kind ap = Direct id -> CS id = Some ci ->
    callee_ok ci (ap_callee ap) ->
    forallb (plain_simple M) (ap_args ap) = true ->
    ap_exn_extra_args ap = [] -> length (ap_args_arity ap) = ci_nparams ci ->
    ap_continuation ap ∉ D -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap) (Apply ap)
  (* A call of code that never returns, each run returning anywhere: the
     second run's continuation is dead. *)
  | rr_call_never (D : list continuation) (ap : apply_expr) (k : continuation)
      (id : code_id) (ci : code_info) :
    call_ok_c ap id ci -> ci_never ci = true ->
    k ∈ D -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap) (Apply (with_continuation ap k))
  (* The same, returning in the second run to a handler [Invalid]. *)
  | rr_call_never_r (D : list continuation) (ap : apply_expr)
      (id : code_id) (ci : code_info) (w : continuation)
      (wps : list variable) (wmd : handler_metadata) :
    call_ok_c ap id ci -> ci_never ci = true ->
    w <> ap_exn_continuation ap -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap)
      (Let_cont_non_recursive w (Handler wps wmd Invalid)
         (Apply (with_continuation ap w)))
  (* The same, returning in the first run to a wrapper. *)
  | rr_call_never_wraps (D : list continuation) (ap : apply_expr)
      (id : code_id) (ci : code_info) (wc : continuation)
      (cps : list variable) (cmd : handler_metadata) (hb : expr)
      (w : continuation) (wps : list variable) (wmd : handler_metadata) :
    call_ok_c ap id ci -> ci_never ci = true ->
    wc <> ap_exn_continuation ap -> w <> ap_exn_continuation ap ->
    ap_exn_continuation ap ∉ D ->
    rr D (Let_cont_non_recursive wc (Handler cps cmd hb) (Apply ap))
      (Let_cont_non_recursive w (Handler wps wmd Invalid)
         (Apply (with_continuation ap w)))
  (* A handler that only the first run binds, which is dead. *)
  | rr_let_cont_never (D : list continuation) (k : continuation)
      (h : continuation_handler) (body body' : expr) :
    k ∉ free_continuations_of_expr body' -> rr (k :: D) body body' ->
    rr D (Let_cont_non_recursive k h body) body'
with rr_handlers : list continuation
    -> list (continuation * continuation_handler)
    -> list (continuation * continuation_handler) -> Prop :=
  | rr_handlers_nil (D : list continuation) : rr_handlers D [] []
  | rr_handlers_cons (D : list continuation) (k : continuation)
      (ps : list variable) (md : handler_metadata) (h h' : expr)
      (hs hs' : list (continuation * continuation_handler)) :
    forallb (plain_variable M) ps = true -> rr D h h' ->
    rr_handlers D hs hs' ->
    rr_handlers D ((k, Handler ps md h) :: hs) ((k, Handler ps md h') :: hs')
with rr_bindings : list static_binding -> list static_binding -> Prop :=
  | rr_bindings_nil : rr_bindings [] []
  | rr_bindings_symbol (b : static_binding) (bs bs' : list static_binding) :
    FlambdaReaperUnboxingRefinesProofs.local_binding M b = true ->
    rr_bindings bs bs' -> rr_bindings (b :: bs) (b :: bs')
  | rr_bindings_deleted (id : code_id) (bs bs' : list static_binding) :
    rr_bindings bs bs' ->
    rr_bindings (Bind_deleted_code id :: bs) (Bind_deleted_code id :: bs')
  | rr_bindings_code (id : code_id) (c c' : code) (ci : code_info)
      (bs bs' : list static_binding) :
    code_frame id c c' ci ->
    rr (code_dead ci c) (code_body c) (code_body c') ->
    rr_bindings bs bs' ->
    rr_bindings (Bind_code id c :: bs) (Bind_code id c' :: bs').

(* The values of the new slots of a field [u] of a representation, the
   old slot holding [v]. *)
Definition slot_holds (vs' : list (value_slot * value)) (u : unboxed_field)
    (v : value) : Prop :=
  match u with
  | Not_unboxed s => lookup s vs' = Some v
  | Unboxed fields => fields_hold vs' fields v
  end.

(* A closure of the first run and the closure of the second, of a set
   whose representation is [r]. *)
Definition crel (r : closure_repr) (v v' : value) : Prop :=
  ∃ (decls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)),
    v = Value_closure_of_set (cr_current r) decls vs
    ∧ v' = Value_closure_of_set (current_slot r) (rename_function_decls r decls)
             vs'
    ∧ lookup (current_slot r) (rename_function_decls r decls)
      = lookup (cr_current r) decls
    ∧ ∀ (w : value_slot) (u : unboxed_field),
        (Field_value_slot w, u) ∈ cr_fields r ->
        ∃ v0 : value, lookup w vs = Some v0 ∧ slot_holds vs' u v0.

Definition env_rel_c (vars vars' : env) : Prop :=
  env_rel M vars vars'
  ∧ ∀ (x : variable) (r : closure_repr) (w : value),
      CH x = Some r -> lookup x vars = Some w ->
      ∃ w' : value, lookup x vars' = Some w' ∧ crel r w w'.

Definition code_rel (id : code_id) (c c' : code) : Prop :=
  ∃ ci : code_info,
    code_frame id c c' ci ∧ rr (code_dead ci c) (code_body c) (code_body c').

Definition statics_rel (S S' : static_env) : Prop :=
  static_values S' = static_values S
  ∧ map fst (static_code S') = map fst (static_code S)
  ∧ ∀ (id : code_id) (c : code),
      lookup id (static_code S) = Some c ->
      ∃ c' : code, lookup id (static_code S') = Some c' ∧ code_rel id c c'.

(* Frames related. A handler holds frames related at the continuations
   not in its [D], and any frames at those in [D], of both runs or of
   one. *)
Inductive cont_rel : cont_value -> cont_value -> Prop :=
  | cont_rel_closure (D : list continuation) (ps : list variable)
      (md : handler_metadata) (h h' : expr) (vars vars' : env)
      (conts conts' : cont_env)
      (rec rec' : list (continuation * continuation_handler)) :
    forallb (plain_variable M) ps = true ->
    rr (without (map fst rec) D) h h' -> env_rel_c vars vars' ->
    (∀ k : continuation,
       k ∉ D -> cont_rel (resolve k conts) (resolve k conts')) ->
    rr_handlers (without (map fst rec) D) rec rec' ->
    cont_rel (Closure (Handler ps md h) vars conts rec)
      (Closure (Handler ps md h') vars' conts' rec')
  | cont_rel_outer (k : continuation) : cont_rel (Outer k) (Outer k).

Definition conts_rel (D : list continuation) (conts conts' : cont_env)
    : Prop :=
  ∀ k : continuation, k ∉ D -> cont_rel (resolve k conts) (resolve k conts').

Definition related (s t : state) : Prop :=
  ∃ D : list continuation,
  rr D (state_expr s) (state_expr t)
  ∧ env_rel_c (state_vars s) (state_vars t)
  ∧ conts_rel D (state_conts s) (state_conts t)
  ∧ statics_rel (state_statics s) (state_statics t)
  ∧ state_heap s = empty_heap ∧ state_heap t = empty_heap
  ∧ state_reads s = state_reads t.

End Related.

Scheme rr_mut := Induction for rr Sort Prop
  with rr_handlers_mut := Induction for rr_handlers Sort Prop
  with rr_bindings_mut := Induction for rr_bindings Sort Prop.
Combined Scheme rr_comb from rr_mut, rr_handlers_mut, rr_bindings_mut.

(* ======================================================================== *)
(* The new slots of a set, and the reads of them                            *)
(* ======================================================================== *)

Lemma eval_value_slots_lookup (vars : env) (l : list (value_slot * simple))
    (vs : list (value_slot * value)) (w : value_slot) (a : simple) :
  eval_value_slots vars l = Some vs -> lookup w l = Some a ->
  ∃ v : value, eval_simple vars a = Some v ∧ lookup w vs = Some v.
Proof.
  revert vs. induction l as [ | [w0 s0] l IH]; intros vs Hl Ha;
    [ discriminate Ha | ].
  cbn [eval_value_slots] in Hl.
  destruct (eval_simple vars s0) as [v0 | ] eqn:Hs0; [ | discriminate Hl ].
  destruct (eval_value_slots vars l) as [vs0 | ] eqn:Hvs0;
    [ | discriminate Hl ].
  cbn in Hl. injection Hl as <-. rewrite lookup_cons in Ha.
  rewrite lookup_cons.
  destruct (Pos.eqb w w0).
  - injection Ha as <-. exists v0. split; [ exact Hs0 | reflexivity ].
  - exact (IH vs0 eq_refl Ha).
Qed.

Lemma eval_value_slots_some (vars : env) (l : list (value_slot * simple)) :
  (∀ (s : value_slot) (a : simple),
     (s, a) ∈ l -> ∃ v : value, eval_simple vars a = Some v) ->
  ∃ vs : list (value_slot * value), eval_value_slots vars l = Some vs.
Proof.
  induction l as [ | [w0 s0] l IH]; intros Hall; [ exists []; reflexivity | ].
  destruct (Hall w0 s0 (or_introl eq_refl)) as (v0 & Hv0).
  destruct IH as (vs & Hvs);
    [ intros s a Hin; exact (Hall s a (or_intror Hin)) | ].
  exists ((w0, v0) :: vs). cbn [eval_value_slots]. rewrite Hv0, Hvs.
  reflexivity.
Qed.

Lemma eval_value_slots_in (vars : env) (l : list (value_slot * simple))
    (vs : list (value_slot * value)) (s : value_slot) (a : simple)
    (v : value) :
  eval_value_slots vars l = Some vs -> NoDup (map fst l) -> (s, a) ∈ l ->
  eval_simple vars a = Some v -> lookup s vs = Some v.
Proof.
  revert vs. induction l as [ | [w0 s0] l IH]; intros vs Hl Hnd Hin Ha;
    [ destruct Hin | ].
  cbn [eval_value_slots] in Hl.
  destruct (eval_simple vars s0) as [v0 | ] eqn:Hs0; [ | discriminate Hl ].
  destruct (eval_value_slots vars l) as [vs0 | ] eqn:Hvs0;
    [ | discriminate Hl ].
  cbn in Hl. injection Hl as <-. cbn [map fst] in Hnd.
  inversion Hnd as [ | w1 l1 Hw0 Hnd' [Heq1 Heq2]]; subst.
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. rewrite Ha in Hs0. injection Hs0 as ->.
    apply lookup_head.
  - rewrite lookup_other; [ exact (IH vs0 eq_refl Hnd' Hin Ha) | ].
    intros <-. apply Hw0. apply in_map_iff. exists (s, a).
    split; [ reflexivity | exact Hin ].
Qed.


(* A read of a value slot of a changed closure: in the first run, the old
   slot of the closure; and the new slots of the second run are read at
   the new function slot. *)
Lemma slot_read_crel (S S' : static_env) (vars : env) (d : named)
    (w : value_slot) (s : simple) (y : variable) (r : closure_repr)
    (v v' r0 : value) :
  slot_read d = Some (w, s) ->
  (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
  lookup y vars = Some v -> crel r v v' -> eval_named S vars d = Some r0 ->
  ∃ vs' : list (value_slot * value),
    (∀ u : unboxed_field,
       (Field_value_slot w, u) ∈ cr_fields r -> slot_holds vs' u r0)
    ∧ ∀ sl : value_slot,
        project_value_slot S' (current_slot r) sl v' = lookup sl vs'.
Proof.
  intros Hd Hs Hy (decls & vs & vs' & -> & -> & Hdecls & Hslots) Hev.
  assert (Hread : lookup (cr_current r) decls <> None
                  ∧ lookup w vs = Some r0).
  { destruct d as [ | | | | w0 s0 | fs w0 s0 | ]; try discriminate Hd;
      cbn [slot_read] in Hd; injection Hd as <- <-;
      destruct Hs as [-> | (k & ->)]; cbn [eval_named eval_simple] in Hev;
      try (rewrite Hy in Hev); cbn in Hev; try discriminate Hev.
    - destruct (lookup (cr_current r) decls) as [code | ] eqn:Hc;
        cbn in Hev; [ | discriminate Hev ].
      split; [ discriminate | exact Hev ].
    - unfold project_value_slot in Hev. cbn in Hev.
      destruct (lookup (cr_current r) decls) as [code | ] eqn:Hc;
        cbn in Hev; [ | discriminate Hev ].
      destruct (Pos.eqb (cr_current r) fs); [ | discriminate Hev ].
      split; [ discriminate | exact Hev ]. }
  destruct Hread as [Hcur Hw].
  exists vs'. split.
  - intros u Hin. destruct (Hslots w u Hin) as (v0 & Hv0 & Hh).
    rewrite Hw in Hv0. injection Hv0 as <-. exact Hh.
  - intros sl. unfold project_value_slot. cbn. rewrite Hdecls.
    destruct (lookup (cr_current r) decls) as [code | ];
      [ | exfalso; exact (Hcur eq_refl) ].
    cbn. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* The [Let]s of [proj_lets] run, each binding its variable to the new
   slot it reads. *)
Lemma proj_lets_run (cs : function_slot) (y : variable)
    (pairs : list (variable * variable * flambda_kind)) :
  ∀ (body : expr) (vars : env) (conts : cont_env) (S : static_env)
    (hp : heap) (rd : read_record),
    NoDup (map (fun '(x, _, _) => x) pairs) ->
    y ∉ map (fun '(x, _, _) => x) pairs ->
    (∀ (x s : variable) (k : flambda_kind),
       (x, s, k) ∈ pairs ->
       ∃ w : value,
         eval_named S vars (Project_value_slot_from cs s (Var y)) = Some w
         ∧ kind_of_value w = k) ->
    ∃ vars1 : env,
      run (length pairs) (State (proj_lets cs (Var y) pairs body) vars conts S
                            hp rd)
      = Step (State body vars1 conts S hp rd)
      ∧ (∀ z : variable,
           z ∉ map (fun '(x, _, _) => x) pairs ->
           lookup z vars1 = lookup z vars)
      ∧ (∀ (x s : variable) (k : flambda_kind) (w : value),
           (x, s, k) ∈ pairs ->
           eval_named S vars (Project_value_slot_from cs s (Var y)) = Some w ->
           lookup x vars1 = Some w).
Proof.
  induction pairs as [ | [[x s] k] pairs IH]; intros body vars conts S hp rd
    Hnd Hy Heval.
  - exists vars. split; [ reflexivity | ]. split; [ reflexivity | ].
    intros x s k w [].
  - cbn [map] in Hnd, Hy.
    inversion Hnd as [ | x0 l0 Hx Hnd' [Heq0 Heq1]]; subst.
    destruct (IH (Let x k (Project_value_slot_from cs s (Var y)) body) vars
                conts S hp rd Hnd')
      as (vars1 & Hrun & Hother & Hbound).
    { intros Hin. apply Hy. right. exact Hin. }
    { intros x1 s1 k1 Hin. exact (Heval x1 s1 k1 (or_intror Hin)). }
    destruct (Heval x s k (or_introl eq_refl)) as (w & Hw & Hk).
    assert (Hw1 : eval_named S vars1 (Project_value_slot_from cs s (Var y))
                  = Some w).
    { rewrite <- Hw. cbn [eval_named eval_simple]. rewrite Hother;
        [ reflexivity | ].
      intros Hin. apply Hy. right. exact Hin. }
    exists ((x, w) :: vars1). split; [ | split ].
    + change (proj_lets cs (Var y) ((x, s, k) :: pairs) body)
        with (proj_lets cs (Var y) pairs
                (Let x k (Project_value_slot_from cs s (Var y)) body)).
      cbn [length]. rewrite <- Nat.add_1_r, run_add, Hrun, run_succ,
        step_let_exec.
      cbn [exec_named]. rewrite Hw1. cbn. rewrite Hk, kind_eqb_refl.
      reflexivity.
    + intros z Hz. rewrite lookup_other
        by (intros ->; apply Hz; left; reflexivity).
      apply Hother. intros Hin. apply Hz. right. exact Hin.
    + intros x' s' k' w' [Heq | Hin] Hw'.
      * injection Heq as -> -> ->. rewrite Hw in Hw'. injection Hw' as <-.
        apply lookup_head.
      * rewrite lookup_other; [ exact (Hbound x' s' k' w' Hin Hw') | ].
        intros ->. apply Hx. apply in_map_iff. exists (x, s', k').
        split; [ reflexivity | exact Hin ].
Qed.

Lemma proj_lets_free (cs : function_slot) (y : simple)
    (pairs : list (variable * variable * flambda_kind)) :
  ∀ body : expr,
    free_continuations_of_expr (proj_lets cs y pairs body)
    = free_continuations_of_expr body
    ∧ binds_statics (proj_lets cs y pairs body) = binds_statics body.
Proof.
  induction pairs as [ | [[x s] k] pairs IH]; intros body;
    [ split; reflexivity | ].
  change (proj_lets cs y ((x, s, k) :: pairs) body)
    with (proj_lets cs y pairs (Let x k (Project_value_slot_from cs s y) body)).
  destruct (IH (Let x k (Project_value_slot_from cs s y) body)) as [H1 H2].
  rewrite H1, H2. split; reflexivity.
Qed.

(* ======================================================================== *)
(* Related environments                                                     *)
(* ======================================================================== *)

Section Simulation.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.
Variable CH : variable -> option closure_repr.
Hypothesis CH_M : ∀ (x : variable) (r : closure_repr),
  CH x = Some r -> fields_of M x = Some [].
Hypothesis CH_fresh : ∀ (x : variable) (r : closure_repr),
  CH x = Some r -> among x (field_variables M) = false.
Variable CS : code_id -> option code_info.

Lemma plain_not_changed (x : variable) :
  plain_variable M x = true -> CH x = None.
Proof.
  intros Hx. destruct (CH x) as [r | ] eqn:E; [ | reflexivity ].
  exfalso. exact (plain_not_unboxed M x [] Hx (CH_M x r E)).
Qed.

Lemma fields_hold_nil (vars' : env) (w : value) : fields_hold vars' [] w.
Proof.
  intros p v k Hp. apply upath_unboxed_inv in Hp as (f & u & p0 & _ & [] & _).
Qed.

(* A variable not among the variables of the fields of an unboxed name
   leaves what they hold as it is. *)
Lemma fields_hold_other (vars' : env) (x : variable) (w' : value)
    (fields : unboxed_fields) (w : value) :
  (∀ (p : list field) (v : variable) (k : flambda_kind),
     upath Value (Unboxed fields) p v k -> v <> x) ->
  fields_hold vars' fields w -> fields_hold ((x, w') :: vars') fields w.
Proof.
  intros Hne Hhold p v k Hp. destruct (Hhold p v k Hp) as (w0 & H1 & H2 & H3).
  exists w0. split; [ exact H1 | split; [ | exact H3 ] ].
  rewrite lookup_other by exact (Hne p v k Hp). exact H2.
Qed.

Lemma env_rel_c_bind (vars vars' : env) (x : variable) (w : value) :
  env_rel_c M CH vars vars' -> plain_variable M x = true ->
  env_rel_c M CH ((x, w) :: vars) ((x, w) :: vars').
Proof.
  intros [Hrel Hch] Hx. split; [ exact (env_rel_bind M _ _ x w Hrel
                                          Hx) | ].
  intros y r w0 Hy Hl.
  assert (Hne : y <> x)
    by (intros ->; rewrite (plain_not_changed x Hx) in Hy; discriminate Hy).
  rewrite lookup_other in Hl |- * by exact Hne. exact (Hch y r w0 Hy Hl).
Qed.

Lemma env_rel_c_bind_changed (vars vars' : env) (x : variable)
    (r : closure_repr) (w w' : value) :
  env_rel_c M CH vars vars' -> CH x = Some r -> crel r w w' ->
  env_rel_c M CH ((x, w) :: vars) ((x, w') :: vars').
Proof.
  intros [[Hplain Hfs] Hch] Hx Hc. split; [ split | ].
  - intros z Hz.
    assert (Hne : z <> x)
      by (intros ->; rewrite (plain_not_changed x Hz) in Hx; discriminate Hx).
    rewrite !lookup_other by exact Hne. exact (Hplain z Hz).
  - intros y fy w0 Hy Hl. destruct (Pos.eq_dec y x) as [-> | Hne].
    + rewrite (CH_M x r Hx) in Hy. injection Hy as <-. apply fields_hold_nil.
    + rewrite lookup_other in Hl by exact Hne.
      apply fields_hold_other; [ | exact (Hfs y fy w0 Hy Hl) ].
      intros p v k Hp ->.
      pose proof (field_variables_in M y fy x Hy
                    (upath_leaf_in fy p x k Hp)) as Hin.
      apply among_In in Hin. rewrite (CH_fresh x r Hx) in Hin.
      discriminate Hin.
  - intros y r' w0 Hy Hl. destruct (Pos.eq_dec y x) as [-> | Hne].
    + rewrite lookup_head in Hl |- *. injection Hl as <-.
      rewrite Hx in Hy. injection Hy as <-. exists w'.
      split; [ reflexivity | exact Hc ].
    + rewrite lookup_other in Hl |- * by exact Hne. exact (Hch y r' w0 Hy Hl).
Qed.

(* The run of the second binds the variables of the fields of [x], and
   nothing else; the first binds [x], an unboxed name. *)
Lemma env_rel_c_bind_fields (vars vars' vars1 : env) (x : variable)
    (fields : unboxed_fields) (w : value) :
  env_rel_c M CH vars vars' -> fields_of M x = Some fields -> CH x = None ->
  (∀ z : variable,
     z ∉ leaf_variables fields -> lookup z vars1 = lookup z vars') ->
  fields_hold vars1 fields w ->
  env_rel_c M CH ((x, w) :: vars) vars1.
Proof.
  intros [Hrel Hch] Hx Hxc Hother Hhold. split.
  - exact (env_rel_bind_fields M M_distinct vars vars' vars1 x fields w Hrel
             Hx Hother Hhold).
  - intros y r w0 Hy Hl.
    assert (Hne : y <> x) by (intros ->; rewrite Hxc in Hy; discriminate Hy).
    rewrite lookup_other in Hl by exact Hne.
    rewrite Hother; [ exact (Hch y r w0 Hy Hl) | ].
    intros Hin. pose proof (field_variables_in M x fields y Hx Hin) as Hfv.
    apply among_In in Hfv. rewrite (CH_fresh y r Hy) in Hfv.
    discriminate Hfv.
Qed.

Lemma env_rel_c_empty : env_rel_c M CH [] [].
Proof.
  split; [ exact (env_rel_empty M) | ]. intros x r w _ Hl. discriminate Hl.
Qed.



(* The new slots of a set related are bound in the second run to what
   the old slots hold in the first, at the paths of their fields. *)
Lemma new_slots_eval (vars vars' : env) (r : closure_repr)
    (vs : list (value_slot * simple)) (vs1 : list (value_slot * value)) :
  env_rel M vars vars' -> Forall (set_field_ok M CH vs) (cr_fields r) ->
  NoDup (map fst (new_slots M r vs)) ->
  eval_value_slots vars vs = Some vs1 ->
  ∃ vs2 : list (value_slot * value),
    eval_value_slots vars' (new_slots M r vs) = Some vs2
    ∧ ∀ (w : value_slot) (u : unboxed_field),
        (Field_value_slot w, u) ∈ cr_fields r ->
        ∃ v0 : value, lookup w vs1 = Some v0 ∧ slot_holds vs2 u v0.
Proof.
  intros Hrel Hok Hnd Hvs1. rewrite Forall_forall in Hok.
  (* What each field makes. *)
  assert (Hfield : ∀ (w : value_slot) (u : unboxed_field),
            (Field_value_slot w, u) ∈ cr_fields r ->
            ∃ (a : simple) (v0 : value),
              lookup w vs = Some a ∧ eval_simple vars a = Some v0
              ∧ lookup w vs1 = Some v0
              ∧ match u with
                | Not_unboxed s =>
                  new_slot M vs (Field_value_slot w, u) = [(s, a)]
                  ∧ eval_simple vars' a = Some v0
                | Unboxed nested =>
                  ∃ (b : variable) (fb : unboxed_fields)
                    (pairs : list (variable * variable * flambda_kind)),
                    new_slot M vs (Field_value_slot w, u)
                    = map (fun '(s, x, _) => (s, Var x)) pairs
                    ∧ unboxed_field_pairs Value (Unboxed nested) (Unboxed fb)
                      = Some pairs
                    ∧ fields_hold vars' fb v0
                end).
  { intros w u Hin. specialize (Hok _ Hin). cbn [set_field_ok] in Hok.
    destruct Hok as (a & Ha & Hu).
    destruct (eval_value_slots_lookup vars vs vs1 w a Hvs1 Ha)
      as (v0 & Hv0 & Hl0).
    exists a, v0. split; [ exact Ha | split; [ exact Hv0 | ] ].
    split; [ exact Hl0 | ]. unfold new_slot. rewrite Ha.
    destruct u as [s | nested].
    - split; [ reflexivity | ].
      rewrite <- (plain_simple_eval M vars vars' a Hrel Hu). exact Hv0.
    - destruct Hu as (b & fb & pairs & -> & _ & Hb & Hp).
      exists b, fb, pairs. rewrite Hb, Hp.
      split; [ reflexivity | split; [ reflexivity | ] ].
      exact (proj2 Hrel b fb v0 Hb Hv0). }
  destruct (eval_value_slots_some vars' (new_slots M r vs)) as (vs2 & Hvs2).
  { intros s a' Hin. unfold new_slots in Hin.
    apply in_flat_map in Hin as ([f u] & Hfu & Hin).
    destruct f as [ | w | | | | | | | ];
      try (cbn [new_slot] in Hin; destruct Hin; fail).
    destruct (Hfield w u Hfu) as (a & v0 & _ & _ & _ & Hcase).
    destruct u as [s0 | nested].
    - destruct Hcase as [Heq Hv']. rewrite Heq in Hin.
      destruct Hin as [Heq' | []]. injection Heq' as <- <-.
      exists v0. exact Hv'.
    - destruct Hcase as (b & fb & pairs & Heq & Hp & Hhold).
      rewrite Heq in Hin. apply in_map_iff in Hin as ([[s1 x] k] & Heq' & Hin).
      injection Heq' as <- <-.
      destruct (pairs_hold vars' nested fb v0 pairs Hp Hhold)
        as (_ & Hseconds & _).
      destruct (Hseconds s1 x k Hin) as (_ & w' & Hl' & _).
      exists w'. exact Hl'. }
  exists vs2. split; [ exact Hvs2 | ].
  intros w u Hin.
  destruct (Hfield w u Hin) as (a & v0 & _ & _ & Hl0 & Hcase).
  exists v0. split; [ exact Hl0 | ].
  assert (Hsub : ∀ e, e ∈ new_slot M vs (Field_value_slot w, u) ->
                   e ∈ new_slots M r vs)
    by (intros e He; unfold new_slots; apply in_flat_map;
        exists (Field_value_slot w, u); split; assumption).
  destruct u as [s | nested]; cbn [slot_holds].
  - destruct Hcase as [Heq Hv'].
    apply (eval_value_slots_in vars' _ vs2 s a v0 Hvs2 Hnd); [ | exact Hv' ].
    apply Hsub. rewrite Heq. left. reflexivity.
  - destruct Hcase as (b & fb & pairs & Heq & Hp & Hhold).
    destruct (pairs_hold vars' nested fb v0 pairs Hp Hhold)
      as (_ & Hseconds & Hpaths).
    intros p s k Hpath.
    destruct (Hpaths p s k Hpath) as (x & w' & Hin' & Hl' & Hpw).
    destruct (Hseconds s x k Hin') as (_ & w'' & Hl'' & Hk).
    rewrite Hl' in Hl''. injection Hl'' as <-.
    exists w'. split; [ exact Hpw | split; [ | exact Hk ] ].
    apply (eval_value_slots_in vars' _ vs2 s (Var x) w' Hvs2 Hnd);
      [ | exact Hl' ].
    apply Hsub. rewrite Heq. apply in_map_iff. exists (s, x, k).
    split; [ reflexivity | exact Hin' ].
Qed.


(* The [Let]s that the unboxing makes of the fields of [x], of simples,
   run, and bind them as the relation asks. *)
Lemma lets_case_c (vars vars' : env) (x : variable) (fields : unboxed_fields)
    (w : value) (l : list (variable * flambda_kind * simple)) (body : expr)
    (conts : cont_env) (S : static_env) (hp : heap) (rcd : read_record) :
  env_rel_c M CH vars vars' -> fields_of M x = Some fields -> CH x = None ->
  map lets_variable l = leaf_variables fields ->
  (∀ (v : variable) (kind : flambda_kind) (s : simple) (v' : variable),
     (v, kind, s) ∈ l -> v' ∈ leaf_variables fields -> s <> Var v') ->
  (∀ (v : variable) (kind : flambda_kind) (s : simple),
     (v, kind, s) ∈ l ->
     ∃ w' : value, eval_simple vars' s = Some w' ∧ kind_of_value w' = kind) ->
  (∀ (p : list field) (v : variable) (k : flambda_kind),
     upath Value (Unboxed fields) p v k ->
     ∃ (s : simple) (w' : value),
       (v, k, s) ∈ l ∧ eval_simple vars' s = Some w'
       ∧ path_value p w = Some w') ->
  ∃ vars1 : env,
    run (length l) (State (lets l body) vars' conts S hp rcd)
    = Step (State body vars1 conts S hp rcd)
    ∧ env_rel_c M CH ((x, w) :: vars) vars1.
Proof.
  intros Hvars Hx Hxc Hvl Hread Heval Hpaths.
  destruct (lets_run l body vars' conts S hp rcd)
    as (vars1 & Hrun & Hother & Hbound).
  - rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hx).
  - intros v kind s v' Hin Hv'. rewrite Hvl in Hv'.
    exact (Hread v kind s v' Hin Hv').
  - exact Heval.
  - exists vars1. split; [ exact Hrun | ].
    apply (env_rel_c_bind_fields vars vars' vars1 x fields w Hvars Hx Hxc).
    + intros z Hz. apply Hother. rewrite Hvl. exact Hz.
    + intros p v k Hp. destruct (Hpaths p v k Hp) as (s & w' & Hin & Hs & Hpw).
      destruct (Heval v k s Hin) as (w'' & Hs' & Hk).
      rewrite Hs in Hs'. injection Hs' as <-.
      exists w'. split; [ exact Hpw | ].
      split; [ exact (Hbound v k s w' Hin Hs) | exact Hk ].
Qed.

(* The [Let]s of [proj_lets] of the fields of [x] run, and bind them as
   the relation asks. *)
Lemma proj_lets_case (vars vars' : env) (x y : variable)
    (fields : unboxed_fields) (w : value) (cs : function_slot)
    (pairs : list (variable * variable * flambda_kind)) (body : expr)
    (conts : cont_env) (S : static_env) (hp : heap) (rcd : read_record) :
  env_rel_c M CH vars vars' -> fields_of M x = Some fields -> CH x = None ->
  map (fun '(a, _, _) => a) pairs = leaf_variables fields ->
  y ∉ leaf_variables fields ->
  (∀ (v s : variable) (k : flambda_kind),
     (v, s, k) ∈ pairs ->
     ∃ w' : value,
       eval_named S vars' (Project_value_slot_from cs s (Var y)) = Some w'
       ∧ kind_of_value w' = k) ->
  (∀ (p : list field) (v : variable) (k : flambda_kind),
     upath Value (Unboxed fields) p v k ->
     ∃ (s : variable) (w' : value),
       (v, s, k) ∈ pairs
       ∧ eval_named S vars' (Project_value_slot_from cs s (Var y)) = Some w'
       ∧ path_value p w = Some w') ->
  ∃ vars1 : env,
    run (length pairs) (State (proj_lets cs (Var y) pairs body) vars' conts S
                          hp rcd)
    = Step (State body vars1 conts S hp rcd)
    ∧ env_rel_c M CH ((x, w) :: vars) vars1.
Proof.
  intros Hvars Hx Hxc Hvl Hy Heval Hpaths.
  destruct (proj_lets_run cs y pairs body vars' conts S hp rcd)
    as (vars1 & Hrun & Hother & Hbound).
  - rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hx).
  - rewrite Hvl. exact Hy.
  - exact Heval.
  - exists vars1. split; [ exact Hrun | ].
    apply (env_rel_c_bind_fields vars vars' vars1 x fields w Hvars Hx Hxc).
    + intros z Hz. apply Hother. rewrite Hvl. exact Hz.
    + intros p v k Hp. destruct (Hpaths p v k Hp) as (s & w' & Hin & Hs & Hpw).
      destruct (Heval v s k Hin) as (w'' & Hs' & Hk).
      rewrite Hs in Hs'. injection Hs' as <-.
      exists w'. split; [ exact Hpw | ].
      split; [ exact (Hbound v s k w' Hin Hs) | exact Hk ].
Qed.

(* ======================================================================== *)
(* Shapes of related terms                                                  *)
(* ======================================================================== *)

Lemma code_free_conts (c : code) :
  free_continuations_of_code c
  = without [code_return_continuation c; code_exn_continuation c]
      (free_continuations_of_expr (code_body c)).
Proof. destruct c. reflexivity. Qed.

(* A free continuation of the second term is one of the first or dead;
   the second binds statics only where the first does. *)
Definition fc_below (D : list continuation) (l l' : list continuation)
    : Prop :=
  ∀ k : continuation, k ∈ l' -> k ∈ l ∨ k ∈ D.

Lemma orb_impl (a b a' b' : bool) :
  (a' = true -> a = true) -> (b' = true -> b = true) ->
  a' || b' = true -> a || b = true.
Proof.
  intros Ha Hb H. apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (Ha H) | right; exact (Hb H) ].
Qed.

Lemma rr_shape :
  (∀ (D : list continuation) (p q : expr),
     rr M CH CS D p q ->
     fc_below D (free_continuations_of_expr p) (free_continuations_of_expr q)
     ∧ (binds_statics q = true -> binds_statics p = true))
  ∧ (∀ (D : list continuation)
       (hs hs' : list (continuation * continuation_handler)),
       rr_handlers M CH CS D hs hs' ->
       map fst hs = map fst hs'
       ∧ fc_below D
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs)
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs')
       ∧ (handlers_bind_statics hs' = true ->
          handlers_bind_statics hs = true))
  ∧ (∀ bs bs' : list static_binding,
       rr_bindings M CH CS bs bs' ->
       incl (flat_map free_continuations_of_binding bs')
         (flat_map free_continuations_of_binding bs)
       ∧ (existsb binding_binds bs' = true ->
          existsb binding_binds bs = true)).
Proof.
  apply (rr_comb M CH CS
    (fun D p q _ =>
       fc_below D (free_continuations_of_expr p) (free_continuations_of_expr q)
       ∧ (binds_statics q = true -> binds_statics p = true))
    (fun D hs hs' _ =>
       map fst hs = map fst hs'
       ∧ fc_below D
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs)
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs')
       ∧ (handlers_bind_statics hs' = true ->
          handlers_bind_statics hs = true))
    (fun bs bs' _ =>
       incl (flat_map free_continuations_of_binding bs')
         (flat_map free_continuations_of_binding bs)
       ∧ (existsb binding_binds bs' = true ->
          existsb binding_binds bs = true)));
    intros; repeat match goal with
                   | H : _ ∧ _ |- _ => destruct H
                   end;
    unfold fc_below in *.
  (* The [Let]s of both, and the reads of new slots. *)
  1, 2, 6, 7, 8: cbn [free_continuations_of_expr binds_statics];
    split; assumption.
  (* The [Let]s that the unboxing makes. *)
  1: destruct (lets_free_continuations (concat ls) body') as [H1 H2].
  2, 3: destruct (lets_free_continuations (pairs_lets pairs) body')
          as [H1 H2].
  4: destruct (proj_lets_free (current_slot r) (Var y) pairs body')
       as [H1 H2].
  1, 2, 3, 4: cbn [free_continuations_of_expr binds_statics];
    rewrite H1, H2; split; assumption.
  (* [Let_static]. *)
  - cbn [free_continuations_of_expr binds_statics].
    split; [ | apply orb_impl; assumption ].
    intros k0 Hk0. apply in_app_or in Hk0 as [Hk0 | Hk0].
    + left. apply in_or_app. left. auto.
    + match goal with
      | H : ∀ k, k ∈ free_continuations_of_expr body' -> _ |- _ =>
        destruct (H k0 Hk0) as [Hk1 | Hk1]
      end; [ left; apply in_or_app; right; exact Hk1 | right; exact Hk1 ].
  (* [Let_cont]. *)
  - cbn [free_continuations_of_expr binds_statics free_continuations_of_handler
         handler_binds_statics].
    split; [ | apply orb_impl; assumption ].
    intros k0 Hk0. apply in_app_or in Hk0 as [Hk0 | Hk0].
    + match goal with
      | H : ∀ k, k ∈ free_continuations_of_expr h' -> _ |- _ =>
        destruct (H k0 Hk0) as [Hk1 | Hk1]
      end; [ left; apply in_or_app; left; exact Hk1 | right; exact Hk1 ].
    + apply in_without_iff in Hk0 as [Hk0 Hnk].
      match goal with
      | H : ∀ k, k ∈ free_continuations_of_expr body' -> _ |- _ =>
        destruct (H k0 Hk0) as [Hk1 | Hk1]
      end.
      * left. apply in_or_app. right. apply in_without_iff.
        split; assumption.
      * right. apply in_without_iff in Hk1 as [Hk1 _]. exact Hk1.
  - cbn [free_continuations_of_expr binds_statics].
    split.
    + intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
      match goal with
      | H : map fst hs = map fst hs' |- _ => rewrite <- H in Hnk
      end.
      apply in_app_or in Hk0 as [Hk0 | Hk0];
        match goal with
        | H : ∀ k, k ∈ ?l -> _ , Hk : k0 ∈ ?l |- _ =>
          destruct (H k0 Hk) as [Hk1 | Hk1]
        end;
        try (right; apply in_without_iff in Hk1 as [Hk1 _]; exact Hk1);
        left; apply in_without_iff; (split; [ | exact Hnk ]);
        apply in_or_app; [ left | right ]; exact Hk1.
    + apply orb_impl; [ | assumption ].
      unfold handlers_bind_statics in *. assumption.
  (* The others are the same in both. *)
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  (* The calls of code that never returns. *)
  - cbn [free_continuations_of_expr binds_statics ap_continuation
         ap_exn_continuation with_continuation].
    split; [ | intros Hb; discriminate Hb ].
    intros k0 [<- | [<- | []]]; [ right; assumption | ].
    left. right. left. reflexivity.
  - cbn [free_continuations_of_expr binds_statics ap_continuation
         ap_exn_continuation with_continuation free_continuations_of_handler
         handler_binds_statics app].
    split; [ | intros Hb; discriminate Hb ].
    intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
    destruct Hk0 as [<- | [<- | []]];
      [ exfalso; apply Hnk; left; reflexivity | ].
    left. right. left. reflexivity.
  - cbn [free_continuations_of_expr binds_statics ap_continuation
         ap_exn_continuation with_continuation free_continuations_of_handler
         handler_binds_statics app].
    split; [ | intros Hb; discriminate Hb ].
    intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
    destruct Hk0 as [<- | [<- | []]];
      [ exfalso; apply Hnk; left; reflexivity | ].
    left. apply in_or_app. right. apply in_without_iff.
    split; [ right; left; reflexivity | ].
    intros [Heq | []]. auto.
  (* A handler that only the first run binds. *)
  - cbn [free_continuations_of_expr binds_statics].
    split; [ | intros Hb; apply orb_true_iff; right; auto ].
    intros k0 Hk0.
    match goal with
    | H : ∀ k1, k1 ∈ free_continuations_of_expr body' -> _ |- _ =>
      destruct (H k0 Hk0) as [Hk1 | [Hk1 | Hk1]]
    end.
    + left. apply in_or_app. right. apply in_without_iff.
      split; [ exact Hk1 | ]. intros [Heq | []]. subst k0. contradiction.
    + subst k0. contradiction.
    + right. exact Hk1.
  (* Handlers. *)
  - split; [ reflexivity | ].
    split; [ intros k0 [] | intros Hb; exact Hb ].
  - cbn [map fst flat_map handlers_bind_statics existsb].
    split; [ f_equal; assumption | ].
    split.
    + intros k0 Hk0. cbn [free_continuations_of_handler] in Hk0 |- *.
      apply in_app_or in Hk0 as [Hk0 | Hk0];
        match goal with
        | H : ∀ k, k ∈ ?l -> _ , Hk : k0 ∈ ?l |- _ =>
          destruct (H k0 Hk) as [Hk1 | Hk1]
        end;
        try (right; exact Hk1); left; apply in_or_app;
        [ left | right ]; exact Hk1.
    + unfold handlers_bind_statics in *. cbn [existsb].
      apply orb_impl; assumption.
  (* Bindings. *)
  - split; [ intros k0 []| intros Hb; exact Hb ].
  - cbn [flat_map existsb].
    split; [ apply incl_app_app; [ apply incl_refl | assumption ] | ].
    apply orb_impl; [ intros Hb; exact Hb | assumption ].
  - cbn [flat_map existsb free_continuations_of_binding binding_binds].
    split; assumption.
  - change (flat_map free_continuations_of_binding (Bind_code id c :: bs))
      with (free_continuations_of_code c
            ++ flat_map free_continuations_of_binding bs).
    change (flat_map free_continuations_of_binding (Bind_code id c' :: bs'))
      with (free_continuations_of_code c'
            ++ flat_map free_continuations_of_binding bs').
    cbn [existsb binding_binds orb].
    split; [ | intros Hb; reflexivity ].
    rewrite !code_free_conts.
    match goal with
    | H : code_frame _ _ _ _ _ _ _ |- _ =>
      destruct H as (_ & _ & _ & Hret & Hexn & _)
    end.
    rewrite Hret, Hexn. apply incl_app_app; [ | assumption ].
    intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
    match goal with
    | H : ∀ k, k ∈ free_continuations_of_expr (code_body c') -> _ |- _ =>
      destruct (H k0 Hk0) as [Hk1 | Hk1]
    end.
    + apply in_without_iff. split; assumption.
    + exfalso. apply Hnk. unfold code_dead in Hk1.
      destruct (ci_never ci); [ | destruct Hk1 ].
      destruct Hk1 as [<- | []]. left. reflexivity.
Qed.

(* The dead continuations count only where free in the second term. *)
Definition dead_agree (D D' : list continuation) (l : list continuation)
    : Prop :=
  ∀ k : continuation, k ∈ l -> k ∈ D <-> k ∈ D'.

Lemma dead_agree_incl (D D' l l' : list continuation) :
  incl l l' -> dead_agree D D' l' -> dead_agree D D' l.
Proof. intros Hi H k Hk. exact (H k (Hi k Hk)). Qed.

Lemma dead_agree_without (D D' bound l : list continuation) :
  dead_agree D D' (without bound l) ->
  dead_agree (without bound D) (without bound D') l.
Proof.
  intros H k Hk. rewrite !in_without_iff.
  split; intros [Hd Hb]; (split; [ | exact Hb ]);
    apply (H k (proj2 (in_without_iff _ _ _) (conj Hk Hb))); exact Hd.
Qed.

Lemma dead_agree_cons (D D' l : list continuation) (k : continuation) :
  dead_agree D D' l -> dead_agree (k :: D) (k :: D') l.
Proof.
  intros H k0 Hk0. split; intros [He | Hin];
    [ left; exact He | right; exact (proj1 (H k0 Hk0) Hin)
    | left; exact He | right; exact (proj2 (H k0 Hk0) Hin) ].
Qed.

Lemma rr_dead_ext :
  (∀ (D : list continuation) (p q : expr),
     rr M CH CS D p q ->
     ∀ D' : list continuation,
       dead_agree D D' (free_continuations_of_expr q) -> rr M CH CS D' p q)
  ∧ (∀ (D : list continuation)
       (hs hs' : list (continuation * continuation_handler)),
       rr_handlers M CH CS D hs hs' ->
       ∀ D' : list continuation,
         dead_agree D D'
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs') ->
         rr_handlers M CH CS D' hs hs')
  ∧ (∀ bs bs' : list static_binding, rr_bindings M CH CS bs bs' -> True).
Proof.
  apply (rr_comb M CH CS
    (fun D p q _ =>
       ∀ D' : list continuation,
         dead_agree D D' (free_continuations_of_expr q) -> rr M CH CS D' p q)
    (fun D hs hs' _ =>
       ∀ D' : list continuation,
         dead_agree D D'
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs') ->
         rr_handlers M CH CS D' hs hs')
    (fun _ _ _ => True)); intros; try exact I.
  (* The [Let]s of both, and the reads of new slots. *)
  1, 2, 6, 7, 8:
    econstructor; try eassumption;
    match goal with
    | IH : ∀ D', dead_agree _ D' _ -> _ |- _ => apply IH
    end;
    assumption.
  (* The [Let]s that the unboxing makes. *)
  1: destruct (lets_free_continuations (concat ls) body') as [H1 _].
  2, 3: destruct (lets_free_continuations (pairs_lets pairs) body')
          as [H1 _].
  4: destruct (proj_lets_free (current_slot r) (Var y) pairs body')
       as [H1 _].
  1, 2, 3, 4:
    econstructor; try eassumption;
    match goal with
    | IH : ∀ D', dead_agree _ D' _ -> _ |- _ => apply IH
    end;
    match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H; rewrite H1 in H; exact H
    end.
  (* [Let_static]. *)
  - match goal with
    | H : dead_agree _ _ _, IH : ∀ D', dead_agree _ D' _ -> _ |- _ =>
      apply rr_static; [ assumption | apply IH ];
      eapply dead_agree_incl; [ | exact H ];
      intros k0 Hk0; apply in_or_app; right; exact Hk0
    end.
  (* [Let_cont]. *)
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr free_continuations_of_handler] in H
    end.
    apply rr_let_cont; [ assumption | | ].
    + match goal with
      | IH : ∀ D', dead_agree D D' (free_continuations_of_expr h') -> _
        |- _ => apply IH
      end.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. left. exact Hk0.
    + match goal with
      | IH : ∀ D', dead_agree (without [k] D) D'
                     (free_continuations_of_expr body') -> _ |- _ => apply IH
      end.
      apply dead_agree_without. eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. right. exact Hk0.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H
    end.
    match goal with
    | Hhs : rr_handlers M CH CS _ hs hs' |- _ =>
      pose proof (proj1 (proj1 (proj2 rr_shape) _ _ _ Hhs)) as Hkeys
    end.
    apply rr_let_cont_rec.
    + match goal with
      | IH : ∀ D', dead_agree _ D' (flat_map _ hs') -> _ |- _ => apply IH
      end.
      rewrite Hkeys. apply dead_agree_without.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hn].
      apply in_without_iff. split; [ | exact Hn ].
      apply in_or_app. left. exact Hk0.
    + match goal with
      | IH : ∀ D', dead_agree _ D' (free_continuations_of_expr body') -> _
        |- _ => apply IH
      end.
      rewrite Hkeys. apply dead_agree_without.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hn].
      apply in_without_iff. split; [ | exact Hn ].
      apply in_or_app. right. exact Hk0.
  (* The others read [D] at continuations free in the second term. *)
  - apply rr_apply_cont; [ assumption | ].
    intros Hin. match goal with
                | H : dead_agree _ _ _ |- _ =>
                  apply (proj2 (H _ (or_introl eq_refl))) in Hin
                end.
    contradiction.
  - apply rr_switch; [ assumption | assumption | ].
    intros arm Harm Hin.
    assert (Hf : ac_continuation (snd arm)
                 ∈ free_continuations_of_expr (Switch scrutinee arms))
      by (cbn [free_continuations_of_expr]; apply in_flat_map;
          exists arm; split; [ exact Harm | left; reflexivity ]).
    match goal with
    | H : dead_agree _ _ _ |- _ => apply (proj2 (H _ Hf)) in Hin
    end.
    match goal with
    | H : ∀ arm, arm ∈ arms -> _ |- _ => exact (H arm Harm Hin)
    end.
  - apply rr_invalid.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H
    end.
    eapply rr_call; try eassumption.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_introl eq_refl))) in Hin
                  end.
      contradiction.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_intror (or_introl eq_refl)))) in Hin
                  end.
      contradiction.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr with_continuation ap_continuation
           ap_exn_continuation] in H
    end.
    apply rr_call_never with (id := id) (ci := ci); try assumption.
    + match goal with
      | H : dead_agree _ _ _ |- _ =>
        apply (proj1 (H _ (or_introl eq_refl))); assumption
      end.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_intror (or_introl eq_refl)))) in Hin
                  end.
      contradiction.
  - apply rr_call_never_r with (id := id) (ci := ci); try assumption.
    intros Hin.
    match goal with
    | H : dead_agree _ _ (free_continuations_of_expr ?q) |- _ =>
      assert (Hf : ap_exn_continuation ap ∈ free_continuations_of_expr q)
        by (cbn [free_continuations_of_expr free_continuations_of_handler app
                 with_continuation ap_continuation ap_exn_continuation];
            apply in_without_iff; split; [ right; left; reflexivity | ];
            intros [He | []]; auto);
      apply (proj2 (H _ Hf)) in Hin
    end.
    contradiction.
  - apply rr_call_never_wraps with (id := id) (ci := ci); try assumption.
    intros Hin.
    match goal with
    | H : dead_agree _ _ (free_continuations_of_expr ?q) |- _ =>
      assert (Hf : ap_exn_continuation ap ∈ free_continuations_of_expr q)
        by (cbn [free_continuations_of_expr free_continuations_of_handler app
                 with_continuation ap_continuation ap_exn_continuation];
            apply in_without_iff; split; [ right; left; reflexivity | ];
            intros [He | []]; auto);
      apply (proj2 (H _ Hf)) in Hin
    end.
    contradiction.
  - apply rr_let_cont_never; [ assumption | ].
    match goal with
    | IH : ∀ D', dead_agree _ D' _ -> _ |- _ => apply IH
    end.
    apply dead_agree_cons. assumption.
  (* Handlers. *)
  - apply rr_handlers_nil.
  - cbn [flat_map free_continuations_of_handler] in *.
    apply rr_handlers_cons; [ assumption | | ].
    + match goal with
      | IH : ∀ D', dead_agree _ D' (free_continuations_of_expr h') -> _
        |- _ => apply IH
      end.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. left. exact Hk0.
    + match goal with
      | IH : ∀ D', dead_agree _ D' (flat_map _ hs') -> _ |- _ => apply IH
      end.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. right. exact Hk0.
Qed.

(* A continuation not free in the second term may be dead. *)
Lemma rr_dead (D : list continuation) (p q : expr) (k : continuation) :
  rr M CH CS D p q -> k ∉ free_continuations_of_expr q ->
  rr M CH CS (k :: D) p q.
Proof.
  intros H Hk. apply (proj1 rr_dead_ext D p q H). intros k0 Hk0.
  split; [ intros Hin; right; exact Hin | ].
  intros [<- | Hin]; [ contradiction | exact Hin ].
Qed.

(* The same, [k] shadowed before. *)
Lemma rr_dead_shadow (D : list continuation) (p q : expr)
    (k : continuation) :
  rr M CH CS (without [k] D) p q -> k ∉ free_continuations_of_expr q ->
  rr M CH CS (k :: D) p q ∧ rr M CH CS D p q.
Proof.
  intros H Hk. split; apply (proj1 rr_dead_ext _ p q H); intros k0 Hk0;
    rewrite in_without_iff;
    (assert (Hne : k0 <> k) by (intros ->; contradiction)).
  - split; [ intros [Hin _]; right; exact Hin | ].
    intros [<- | Hin]; [ contradiction | ].
    split; [ exact Hin | intros [He | []]; exact (Hne (eq_sym He)) ].
  - split; [ intros [Hin _]; exact Hin | ].
    intros Hin.
    split; [ exact Hin | intros [He | []]; exact (Hne (eq_sym He)) ].
Qed.

(* ======================================================================== *)
(* Statics and continuations                                                *)
(* ======================================================================== *)

Lemma closure_of_values (S S' : static_env) (v : value) :
  static_values S' = static_values S -> closure_of S' v = closure_of S v.
Proof.
  intros Hv. destruct v; simpl; try reflexivity. rewrite Hv. reflexivity.
Qed.

Lemma lookup_none_keys {A B : Type} (l : list (positive * A))
    (l' : list (positive * B)) (k : positive) :
  map fst l' = map fst l -> lookup k l = None -> lookup k l' = None.
Proof.
  revert l'. induction l as [ | [k0 a] l IH]; intros [ | [k0' b] l'] Hk Hl;
    try discriminate Hk; [ reflexivity | ].
  injection Hk as -> Hk. rewrite lookup_cons in Hl. rewrite lookup_cons.
  destruct (Pos.eqb k k0); [ discriminate Hl | exact (IH l' Hk Hl) ].
Qed.

Lemma bindings_eval_rel (bs bs' : list static_binding) :
  rr_bindings M CH CS bs bs' ->
  ∀ (vars vars' : env) (S S' S1 : static_env),
    env_rel_c M CH vars vars' -> statics_rel M CH CS S S' ->
    eval_static_bindings vars bs S = Some S1 ->
    ∃ S1' : static_env,
      eval_static_bindings vars' bs' S' = Some S1'
      ∧ statics_rel M CH CS S1 S1'.
Proof.
  induction 1 as [ | b bs bs' Hb _ IH | id bs bs' _ IH
                 | id c c' ci bs bs' Hframe Hbody _ IH];
    intros vars vars' S S' S1 Hvars (Hval & Hkeys & Hcode) Heval.
  - simpl in Heval. injection Heval as <-. exists S'.
    split; [ reflexivity | split; [ exact Hval | split; assumption ] ].
  - destruct b as [ | | sym [ | | tag fields | | | ]]; simpl in Hb;
      try discriminate Hb.
    simpl in Heval |- *. rewrite Hval.
    destruct (lookup sym (static_values S)); [ discriminate Heval | ].
    simpl in Heval |- *.
    rewrite <- (plain_simples_eval M vars vars' fields (proj1 Hvars) Hb).
    destruct (eval_simples vars fields) as [vs | ]; [ | discriminate Heval ].
    simpl in Heval |- *.
    refine (IH vars vars' _
              (Static_env (static_code S')
                 ((sym, Static_block tag vs) :: static_values S))
              S1 Hvars _ Heval).
    split; [ reflexivity | ].
    split; [ exact Hkeys | exact Hcode ].
  - simpl in Heval |- *.
    exact (IH vars vars' S S' S1 Hvars (conj Hval (conj Hkeys Hcode)) Heval).
  - simpl in Heval |- *.
    destruct (lookup id (static_code S)) eqn:Hl; [ discriminate Heval | ].
    rewrite (lookup_none_keys _ _ id Hkeys Hl).
    refine (IH vars vars' _
              (Static_env ((id, c') :: static_code S') (static_values S'))
              S1 Hvars _ Heval).
    split; [ exact Hval | split ].
    + simpl. rewrite Hkeys. reflexivity.
    + intros id0 c0 Hc0. cbn [static_code] in Hc0 |- *.
      rewrite lookup_cons in Hc0. rewrite lookup_cons.
      destruct (Pos.eqb id0 id) eqn:Hid.
      * apply Pos.eqb_eq in Hid. subst id0.
        injection Hc0 as <-. exists c'. split; [ reflexivity | ].
        exists ci. split; assumption.
      * exact (Hcode id0 c0 Hc0).
Qed.

Lemma resolve_rel (D : list continuation) (conts conts' : cont_env)
    (k : continuation) :
  conts_rel M CH CS D conts conts' -> k ∉ D ->
  cont_rel M CH CS (resolve k conts) (resolve k conts').
Proof. intros H Hk. exact (H k Hk). Qed.

Lemma lookup_rec_closures (all hs : list (continuation * continuation_handler))
    (vars : env) (conts : cont_env) (k : continuation) :
  lookup k (map (fun '(k0, h) => (k0, Closure h vars conts all)) hs)
  = option_map (fun h => Closure h vars conts all) (lookup k hs).
Proof.
  induction hs as [ | [k0 h] hs IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k0); [ reflexivity | exact IH ].
Qed.

Lemma resolve_rec (hs : list (continuation * continuation_handler))
    (vars : env) (conts : cont_env) (k : continuation) :
  resolve k (rec_closures hs vars conts ++ conts)
  = match lookup k hs with
    | Some h => Closure h vars conts hs
    | None => resolve k conts
    end.
Proof.
  unfold resolve, rec_closures. rewrite lookup_app, lookup_rec_closures.
  destruct (lookup k hs); reflexivity.
Qed.

Lemma rr_handlers_lookup (D : list continuation)
    (hs hs' : list (continuation * continuation_handler)) (k : continuation) :
  rr_handlers M CH CS D hs hs' ->
  match lookup k hs, lookup k hs' with
  | Some (Handler ps md h), Some (Handler ps' md' h') =>
    ps' = ps ∧ md' = md ∧ forallb (plain_variable M) ps = true
    ∧ rr M CH CS D h h'
  | None, None => True
  | _, _ => False
  end.
Proof.
  induction 1 as [ | D k0 ps md h h' hs hs' Hps Hh _ IH]; [ exact I | ].
  rewrite !lookup_cons. destruct (Pos.eqb k k0); [ | exact IH ].
  split; [ reflexivity | split; [ reflexivity | split; assumption ] ].
Qed.

Lemma rec_closures_rel (D : list continuation)
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  rr_handlers M CH CS (without (map fst handlers) D) handlers handlers' ->
  env_rel_c M CH vars vars' -> conts_rel M CH CS D conts conts' ->
  conts_rel M CH CS (without (map fst handlers) D)
    (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hh Hvars Hconts k Hk. rewrite !resolve_rec.
  pose proof (rr_handlers_lookup _ _ _ k Hh) as Hl.
  destruct (lookup k handlers) as [[ps md h] | ] eqn:E1,
    (lookup k handlers') as [[ps' md' h'] | ] eqn:E2; try contradiction.
  - destruct Hl as (-> & -> & Hps & Hhr).
    exact (cont_rel_closure M CH CS D ps md h h' vars vars' conts conts'
             handlers handlers' Hps Hhr Hvars Hconts Hh).
  - apply Hconts. intros HkD. apply Hk. apply in_without_iff.
    split; [ exact HkD | ]. intros Hin.
    exact (lookup_keys k handlers Hin E1).
Qed.

Lemma bind_params_rel (ps : list variable) :
  ∀ (vs : list value) (vars vars' hv : env),
    forallb (plain_variable M) ps = true -> env_rel_c M CH vars vars' ->
    bind_params ps vs vars = Some hv ->
    ∃ hv' : env, bind_params ps vs vars' = Some hv' ∧ env_rel_c M CH hv hv'.
Proof.
  induction ps as [ | p ps IH]; intros [ | v vs] vars vars' hv Hps Hvars Hb;
    try discriminate Hb.
  - simpl in Hb |- *. injection Hb as <-. exists vars'.
    split; [ reflexivity | exact Hvars ].
  - simpl in Hps, Hb |- *. apply andb_true_iff in Hps as [Hp Hps].
    exact (IH vs _ _ hv Hps (env_rel_c_bind vars vars' p v Hvars Hp) Hb).
Qed.

(* A continuation applied in both runs to the same values. *)
Lemma cont_answered (c c' : cont_value) (vs : list value)
    (S S' : static_env) (rd : read_record) :
  cont_rel M CH CS c c' -> statics_rel M CH CS S S' ->
  match apply_cont_value c vs S empty_heap rd with
  | Undefined => True
  | Exit k ws hp rd1 =>
    apply_cont_value c' vs S' empty_heap rd = Exit k ws hp rd1
  | Step s1 =>
    ∃ t1 : state,
      apply_cont_value c' vs S' empty_heap rd = Step t1
      ∧ related M CH CS s1 t1
  end.
Proof.
  intros Hc Hst.
  destruct Hc as [D ps md h h' vars vars' conts conts' rec rec' Hps Hh Hvars
                    Hconts Hrec | k];
    cbn [apply_cont_value]; [ | reflexivity ].
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds md));
    [ | exact I ].
  destruct (values_of_subkinds S vs (hm_params_subkinds md)) eqn:Hsk;
    [ | exact I ].
  rewrite (values_of_subkinds_same_values S' S vs _ (proj1 Hst)), Hsk.
  destruct (bind_params ps vs vars) as [hv | ] eqn:Hb; [ | exact I ].
  destruct (bind_params_rel ps vs vars vars' hv Hps Hvars Hb)
    as (hv' & Hb' & Hrel).
  rewrite Hb'. eexists. split; [ reflexivity | ].
  exists (without (map fst rec) D).
  split; [ exact Hh | split; [ exact Hrel | ] ].
  split; [ exact (rec_closures_rel D rec rec' vars vars' conts conts' Hrec
                    Hvars Hconts) | ].
  split; [ exact Hst | repeat split ].
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Lemma named_plain_exec_c (S S' : static_env) (hp : heap) (rcd : read_record)
    (vars vars' : env) (d : named) :
  static_values S' = static_values S -> env_rel M vars vars' ->
  FlambdaReaperUnboxingRefinesProofs.named_plain M d = true ->
  exec_named S hp rcd vars d = exec_named S' hp rcd vars' d.
Proof.
  intros Hst Hrel Hd.
  destruct d as [s | p args | | | | | ]; simpl in Hd; try discriminate Hd.
  - simpl. rewrite (plain_simple_eval M vars vars' s Hrel Hd). reflexivity.
  - rewrite (exec_named_static_values S S' hp rcd vars p args)
      by (intros vs0 sym _ _; rewrite Hst; reflexivity).
    simpl. rewrite (plain_simples_eval M vars vars' args Hrel Hd).
    reflexivity.
Qed.

Lemma plain_slots_eval (vars vars' : env) (vs : list (value_slot * simple)) :
  env_rel M vars vars' ->
  forallb (fun ws => plain_simple M (snd ws)) vs = true ->
  eval_value_slots vars vs = eval_value_slots vars' vs.
Proof.
  intros Hrel. induction vs as [ | [w s] vs IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hs H].
  rewrite (plain_simple_eval M vars vars' s Hrel Hs), (IH H). reflexivity.
Qed.

Lemma slot_read_var (S : static_env) (vars : env) (d : named) (w : value_slot)
    (s : simple) (r0 : value) :
  slot_read d = Some (w, s) -> eval_named S vars d = Some r0 ->
  ∃ v : value, eval_simple vars s = Some v.
Proof.
  intros Hd Hev.
  destruct d as [ | | | | w0 s0 | fs w0 s0 | ]; try discriminate Hd;
    cbn [slot_read] in Hd; injection Hd as -> ->; cbn [eval_named] in Hev;
    destruct (eval_simple vars s) as [v | ]; try discriminate Hev;
    exists v; reflexivity.
Qed.

Lemma slot_read_not_heap (d : named) (w : value_slot) (s : simple) :
  slot_read d = Some (w, s) ->
  ∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args.
Proof. intros H q args ->. discriminate H. Qed.

Lemma slot_read_const (S : static_env) (vars : env) (d : named)
    (w : value_slot) (c : const) :
  slot_read d = Some (w, Const c) -> eval_named S vars d = None.
Proof.
  intros Hd. destruct d as [ | | | | w0 s0 | fs w0 s0 | ]; try discriminate Hd;
    cbn [slot_read] in Hd; injection Hd as -> ->; reflexivity.
Qed.

Lemma eval_proj_from (S : static_env) (vars : env) (cs : function_slot)
    (sl : value_slot) (y : variable) (v : value) :
  lookup y vars = Some v ->
  eval_named S vars (Project_value_slot_from cs sl (Var y))
  = project_value_slot S cs sl v.
Proof. intros H. cbn [eval_named eval_simple]. rewrite H. reflexivity. Qed.

Lemma exec_named_proj (S : static_env) (hp : heap) (rd : read_record)
    (vars : env) (cs : function_slot) (sl : value_slot) (y : simple) :
  exec_named S hp rd vars (Project_value_slot_from cs sl y)
  = match eval_named S vars (Project_value_slot_from cs sl y) with
    | Some v => Some (v, hp, rd)
    | None => None
    end.
Proof. reflexivity. Qed.

Lemma find_arm_in (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac -> ∃ d : Z, (d, ac) ∈ arms.
Proof.
  induction arms as [ | [d ac0] arms IH]; intros H; [ discriminate H | ].
  simpl in H. destruct (Z.eqb i d).
  - injection H as <-. exists d. left. reflexivity.
  - destruct (IH H) as (d' & Hin). exists d'. right. exact Hin.
Qed.

(* The values of a callee and of the [my_closure] of the code it calls:
   the same, or, for changed ones, closures related. *)
Definition class_rel (o : option closure_repr) (v v' : value) : Prop :=
  match o with
  | None => v' = v
  | Some r => crel r v v'
  end.

Lemma closure_of_class (o : option closure_repr) (S S' : static_env)
    (v v' : value) (code : code_id) (sl : list (value_slot * value)) :
  class_rel o v v' -> static_values S' = static_values S ->
  closure_of S v = Some (code, sl) ->
  ∃ sl' : list (value_slot * value), closure_of S' v' = Some (code, sl').
Proof.
  intros Hrel Hst Hc. destruct o as [r | ]; cbn [class_rel] in Hrel.
  - destruct Hrel as (decls & vs & vs' & -> & -> & Hdecls & _).
    cbn in Hc |- *. rewrite Hdecls.
    destruct (lookup (cr_current r) decls) as [code' | ];
      [ | discriminate Hc ].
    cbn in Hc |- *. injection Hc as <- _. exists vs'. reflexivity.
  - subst v'. exists sl. rewrite (closure_of_values S S' v Hst). exact Hc.
Qed.

Lemma callee_eval (ci : code_info) (s : simple) (vars vars' : env)
    (v : value) :
  env_rel_c M CH vars vars' -> callee_ok M CH ci s ->
  eval_simple vars s = Some v ->
  ∃ v' : value,
    eval_simple vars' s = Some v' ∧ class_rel (ci_my_closure ci) v v'.
Proof.
  intros [Hrel Hch] Hok Hv. unfold callee_ok in Hok.
  destruct (ci_my_closure ci) as [r | ]; cbn [class_rel].
  - destruct Hok as (f & -> & Hf). simpl in Hv |- *.
    exact (Hch f r v Hf Hv).
  - exists v. rewrite <- (plain_simple_eval M vars vars' s Hrel Hok).
    split; [ exact Hv | reflexivity ].
Qed.

Lemma my_closure_env (ci : code_info) (z : variable) (v v' : value) :
  my_closure_ok M CH ci z -> class_rel (ci_my_closure ci) v v' ->
  env_rel_c M CH [(z, v)] [(z, v')].
Proof.
  unfold my_closure_ok. destruct (ci_my_closure ci) as [r | ];
    cbn [class_rel]; intros Hz Hrel.
  - exact (env_rel_c_bind_changed [] [] z r v v' env_rel_c_empty Hz Hrel).
  - subst v'. exact (env_rel_c_bind [] [] z v env_rel_c_empty Hz).
Qed.

Lemma without_nil (l : list continuation) : without [] l = l.
Proof.
  unfold without. induction l as [ | a l IH]; [ reflexivity | ].
  cbn [filter]. rewrite IH. reflexivity.
Qed.

Lemma with_continuation_same (ap : apply_expr) :
  with_continuation ap (ap_continuation ap) = ap.
Proof. destruct ap. reflexivity. Qed.

Lemma resolve_other (k k0 : continuation) (c : cont_value)
    (conts : cont_env) :
  k <> k0 -> resolve k ((k0, c) :: conts) = resolve k conts.
Proof.
  intros Hk. unfold resolve. rewrite lookup_cons.
  rewrite (proj2 (Pos.eqb_neq k k0) Hk). reflexivity.
Qed.

Lemma run_two (t t1 t2 : state) :
  step t = Step t1 -> step t1 = Step t2 -> run 2 t = Step t2.
Proof.
  intros H1 H2.
  change (run 2 t) with (match step t with Step s' => run 1 s' | r => r end).
  rewrite H1.
  change (run 1 t1) with (match step t1 with Step s' => run 0 s' | r => r end).
  rewrite H2. reflexivity.
Qed.

(* A call of [rr_call], the second run returning to [k2]: both runs enter
   related code, the frames of their return continuations unrelated. *)
Lemma call_entered (ap : apply_expr) (k2 : continuation) (id : code_id)
    (ci : code_info) (vars vars' : env) (conts conts' : cont_env)
    (S S' : static_env) (rcd : read_record) :
  call_ok_c M CH CS ap id ci ->
  env_rel_c M CH vars vars' -> statics_rel M CH CS S S' ->
  match step (State (Apply ap) vars conts S empty_heap rcd) with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ (cc cc' : code) (hv hv' : env),
      s1 = State (code_body cc) hv
             [(code_return_continuation cc,
               resolve (ap_continuation ap) conts);
              (code_exn_continuation cc,
               resolve (ap_exn_continuation ap) conts)]
             S empty_heap rcd
      ∧ step (State (Apply (with_continuation ap k2)) vars' conts' S'
                empty_heap rcd)
        = Step (State (code_body cc') hv'
                  [(code_return_continuation cc, resolve k2 conts');
                   (code_exn_continuation cc,
                    resolve (ap_exn_continuation ap) conts')]
                  S' empty_heap rcd)
      ∧ env_rel_c M CH hv hv'
      ∧ rr M CH CS (code_dead ci cc) (code_body cc) (code_body cc')
  end.
Proof.
  intros (Hkind & Hci & Hcallee & Hargs & Hextra & Hlen) Hvars Hst.
  rewrite (step_apply_no_extra_args ap vars conts S empty_heap rcd Hextra).
  destruct (eval_simple vars (ap_callee ap)) as [c1 | ] eqn:Hc1;
    [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
    [ | exact I ].
  destruct (callee_eval ci (ap_callee ap) vars vars' c1 Hvars Hcallee Hc1)
    as (c2 & Hc2 & Hcl).
  unfold call_function, run_call, shape_of_call. rewrite Hkind.
  cbn beta iota. unfold shape_of_closure_call.
  destruct (closure_of S c1) as [[called sl1] | ] eqn:Hcl1; [ | exact I ].
  destruct (closure_of_class _ S S' c1 c2 called sl1 Hcl (proj1 Hst) Hcl1)
    as (sl2 & Hcl2).
  destruct (Pos.eqb called id) eqn:Hid; [ | exact I ].
  apply Pos.eqb_eq in Hid. subst called.
  destruct (lookup id (static_code S)) as [cc | ] eqn:Hcc; [ | exact I ].
  destruct (proj2 (proj2 Hst) id cc Hcc)
    as (cc' & Hcc' & ci' & Hframe & Hcbody).
  destruct Hframe as (Hci' & Hps & Hmc & Hret & Hexn & Hpa & Hra & Htup
                      & Htup' & Hlenc & Hplain & Hmcok & Hsk').
  rewrite Hci in Hci'. injection Hci' as <-.
  unfold arguments_for_code. rewrite Htup. cbn beta iota.
  unfold shape_of_code_call. rewrite Hlen, <- Hlenc, Nat.eqb_refl.
  cbn beta iota. unfold call_code.
  destruct (binds_statics (code_body cc)) eqn:Hbs; [ exact I | ].
  assert (Hbs' : binds_statics (code_body cc') = false).
  { destruct (binds_statics (code_body cc')) eqn:E; [ | reflexivity ].
    rewrite (proj2 (proj1 rr_shape _ _ _ Hcbody) E) in Hbs.
    discriminate Hbs. }
  match goal with
  | |- context [if negb ?t then Undefined else _] =>
    destruct t eqn:Htest
  end; cbn [negb]; [ | exact I ].
  destruct (values_of_subkinds S vs (code_params_subkinds cc));
    cbn [negb]; [ | exact I ].
  destruct (bind_params (code_params cc) vs [(code_my_closure cc, c1)])
    as [hv | ] eqn:Hb; [ | exact I ].
  destruct (bind_params_rel (code_params cc) vs [(code_my_closure cc, c1)]
              [(code_my_closure cc, c2)] hv Hplain
              (my_closure_env ci (code_my_closure cc) c1 c2 Hmcok Hcl) Hb)
    as (hv' & Hb' & Hrel).
  exists cc, cc', hv, hv'. split; [ reflexivity | ].
  split; [ | split; [ exact Hrel | exact Hcbody ] ].
  rewrite (step_apply_no_extra_args (with_continuation ap k2) vars' conts' S'
             empty_heap rcd Hextra).
  cbn [with_continuation ap_callee ap_call_kind ap_args ap_continuation
       ap_exn_continuation ap_args_arity ap_return_arity].
  rewrite Hc2, <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hargs), Hvs.
  unfold call_function, run_call, shape_of_call. rewrite Hkind.
  cbn beta iota. unfold shape_of_closure_call.
  rewrite Hcl2, Pos.eqb_refl, Hcc'.
  unfold arguments_for_code. rewrite Htup'. cbn beta iota.
  unfold shape_of_code_call. rewrite Hpa, Hlen, <- Hlenc, Nat.eqb_refl.
  cbn beta iota. unfold call_code.
  rewrite Hbs', Hpa, Hra, Htest.
  cbn [negb]. rewrite (values_of_subkinds_nothing S' _ vs Hsk').
  cbn [negb]. rewrite Hps, Hmc, Hb', Hret, Hexn. reflexivity.
Qed.

(* The frames of code entered: those of its exception continuation
   related, and those of its return continuation if it may return. *)
Lemma code_conts_rel (ci : code_info) (cc : code) (r r' x x' : cont_value) :
  (ci_never ci = false -> cont_rel M CH CS r r') -> cont_rel M CH CS x x' ->
  conts_rel M CH CS (code_dead ci cc)
    [(code_return_continuation cc, r); (code_exn_continuation cc, x)]
    [(code_return_continuation cc, r'); (code_exn_continuation cc, x')].
Proof.
  intros Hr Hx k Hk. unfold resolve. rewrite !lookup_cons.
  destruct (Pos.eqb k (code_return_continuation cc)) eqn:E.
  - apply Pos.eqb_eq in E. subst k. apply Hr. unfold code_dead in Hk.
    destruct (ci_never ci); [ | reflexivity ].
    exfalso. apply Hk. left. reflexivity.
  - destruct (Pos.eqb k (code_exn_continuation cc)); [ exact Hx | ].
    constructor.
Qed.

Lemma related_answered (s t : state) :
  related M CH CS s t -> answered (related M CH CS) eq measure s t.
Proof.
  destruct s as [p vars conts S hp rcd],
    t as [q vars' conts' S' hp' rcd'].
  intros (D & Hrr & Hvars & Hconts & Hst & Hhp & Hhp' & Hrcd).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst hp hp' rcd'. unfold answered.
  destruct Hrr as
    [D0 x kind d body body' Hx Hd Hnh Hbody
    | D0 x kind slot decls vs body body' Hx Hvs Hbody
    | D0 x d fields ls body body' Hfields Hxc Hxf Hl Hoff Hbody
    | D0 x y fields fields_y pairs body body' Hfields Hxc Hxf Hy Hne Hl
        Hbody
    | D0 x d f s y fields fields_y nested pairs body body' Hfields Hxc Hxf
        Hread Hs Hy Hne Hin Hl Hbody
    | D0 x kind d f s y v fields body body' Hx Hread Hs Hfields Hin Hbody
    | D0 x r decls vs body body' Hxr Hdecls Hok Hnd Hbody
    | D0 x kind d w s y r sl body body' Hx Hd Hs Hyr Hin Hbody
    | D0 x d w s y r fields nested pairs body body' Hfields Hxc Hxf Hd Hs
        Hyr Hin Hl Hbody
    | D0 bs bs' body body' Hbs Hbody
    | D0 k ps md h h' body body' Hps Hh Hbody
    | D0 hs hs' body body' Hhs Hbody
    | D0 ac Hac Hk
    | D0 scrutinee arms Hscrutinee Harms Hks
    | D0
    | D0 ap id ci Hkind Hci Hcallee Hargs Hextra Hlen Hk Hke
    | D0 ap k id ci Hok Hnv Hk Hke
    | D0 ap id ci w wps wmd Hok Hnv Hw Hke
    | D0 ap id ci wc cps cmd hb w wps wmd Hok Hnv Hwc Hw Hke
    | D0 k h body body' Hnk Hbody ].
  - (* A [Let] of both. *)
    rewrite step_let_exec,
      (named_plain_exec_c S S' empty_heap rcd vars vars' d (proj1 Hst)
         (proj1 Hvars) Hd).
    destruct (exec_named S' empty_heap rcd vars' d)
      as [[[w hp1] rcd1] | ] eqn:Hw; [ | exact I ].
    destruct (exec_named_empty_heap S' rcd vars' d w hp1 rcd1 Hnh Hw)
      as (_ & -> & ->).
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hk; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_exec, Hw, Hk; reflexivity | ].
    exists D0; split; [ exact Hbody | ].
    split; [ exact (env_rel_c_bind _ _ x w Hvars Hx) | ].
    split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* A plain set of closures. *)
    rewrite step_let_exec. cbn [exec_named eval_named].
    rewrite (plain_slots_eval vars vars' vs (proj1 Hvars) Hvs).
    destruct (lookup slot decls) as [code | ] eqn:Hcode; [ | exact I ].
    destruct (eval_value_slots vars' vs) as [vs1 | ] eqn:Hvs1; [ | exact I ].
    cbn. destruct (kind_eqb Value kind) eqn:Hk; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_exec; cbn [exec_named eval_named];
             rewrite Hcode, Hvs1; cbn; rewrite Hk; reflexivity | ].
    exists D0; split; [ exact Hbody | ].
    split; [ exact (env_rel_c_bind _ _ x _ Hvars Hx) | ].
    split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* The allocation of an unboxed variable. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[w hp1] rcd1] | ]
      eqn:Hxw; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d w hp1 rcd1 Hoff Hxw)
      as (Hw & -> & ->).
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (lets_case_c vars vars' x fields w (concat ls) body' conts' S'
                empty_heap rcd Hvars Hfields Hxc)
      as (vars1 & Hrun & Hrel).
    + clear -Hl. induction Hl as [ | [f u] l fields ls Hfu _ IH];
        [ reflexivity | ].
      cbn [concat]. rewrite map_app, IH. unfold leaf_variables.
      rewrite leaves_cons, map_app. f_equal.
      destruct u as [v | nested]; cbn [alloc_entry] in Hfu.
      * destruct Hfu as (kind & s & -> & _). reflexivity.
      * destruct Hfu as (b & fb & pairs & _ & _ & _ & Hpairs & ->).
        rewrite pairs_lets_variables.
        destruct (pairs_spec _ _ _ _ Hpairs) as (_ & _ & Hc & _).
        rewrite Hc. reflexivity.
    + intros v kind s v' Hin Hv' ->.
      destruct (forall2_concat_in _ _ _ _ Hl Hin)
        as ([f u] & l0 & _ & Hent & Hin0).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * destruct Hent as (kind0 & s0 & -> & _ & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- ->.
        simpl in Hplain. apply plain_variable_iff in Hplain as [_ Hnot].
        pose proof (field_variables_in M x fields v' Hfields Hv') as Hfv.
        apply among_In in Hfv. rewrite Hfv in Hnot. discriminate.
      * destruct Hent as (b & fb & pairs & _ & Hb & Hxb & Hpairs & ->).
        apply in_pairs_lets in Hin0 as (v'' & Heq & Hin').
        injection Heq as <-.
        destruct (pairs_spec _ _ _ _ Hpairs) as (Ha & _).
        destruct (Ha v v' kind Hin') as (p & _ & Hp').
        exact (leaves_apart M M_distinct b x fb fields v' Hb Hfields
                 (not_eq_sym Hxb) (upath_leaf_in fb p v' kind Hp') Hv').
    + intros v kind s Hin.
      destruct (forall2_concat_in _ _ _ _ Hl Hin)
        as ([f u] & l0 & _ & Hent & Hin0).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- <-.
        destruct (alloc_field_eval S vars d w f kind0 s0 Hw Hf)
          as (w' & Hs & _ & Hkw).
        exists w'. split; [ | exact Hkw ].
        rewrite <- (plain_simple_eval M vars vars' s0 (proj1 Hvars) Hplain).
        exact Hs.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval S vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & _ & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 (proj1 Hvars) b fb wb Hb Hs))
          as (_ & Hseconds & _).
        apply in_pairs_lets in Hin0 as (v' & -> & Hin').
        destruct (Hseconds v v' kind Hin') as [_ (w' & Hl' & Hk')].
        exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k Hpath.
      apply upath_unboxed_inv in Hpath as (f & u & p0 & -> & Hfu & Hp0).
      destruct (forall2_in_both _ _ _ _ Hl Hfu) as (l0 & Hl0 & Hent).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * apply upath_not_unboxed_inv in Hp0 as (-> & -> & ->).
        destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        pose proof (alloc_field_kind d f kind0 s0 Hf) as Hk0. subst kind0.
        destruct (alloc_field_eval S vars d w f (field_kind f) s0 Hw Hf)
          as (w' & Hs & Hfw & _).
        exists s0, w'. split; [ exact (in_concat_of _ _ _ Hl0 (or_introl
                                                       eq_refl)) | ].
        split; [ rewrite <- (plain_simple_eval M vars vars' s0 (proj1 Hvars)
                               Hplain);
                 exact Hs | ].
        cbn [path_value]. rewrite Hfw. reflexivity.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval S vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & Hfw & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 (proj1 Hvars) b fb wb Hb Hs))
          as (_ & _ & Hpaths).
        destruct (Hpaths p0 v k (upath_unboxed_kind _ Value _ _ _ _ Hp0))
          as (v' & w' & Hin' & Hl' & Hpw).
        exists (Var v'), w'.
        split; [ exact (in_concat_of _ _ _ Hl0
                          (pairs_in_lets pairs v v' k Hin')) | ].
        split; [ exact Hl' | ]. cbn [path_value]. rewrite Hfw. exact Hpw.
    + apply (run_answer (related M CH CS) _ _ _ _ (length (concat ls)) Hrun).
      * exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* An alias of an unboxed variable. *)
    rewrite step_let_simple. cbn [eval_simple].
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (pairs_hold vars' fields fields_y w pairs Hl
                (proj2 (proj1 Hvars) y fields_y w Hy Hw))
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case_c vars vars' x fields w (pairs_lets pairs) body'
                conts' S' empty_heap rcd Hvars Hfields Hxc)
      as (vars1 & Hrun & Hrel).
    + rewrite pairs_lets_variables. exact Hfirsts.
    + intros v kind s v' Hin Hv' ->.
      apply in_pairs_lets in Hin as (v'' & Heq & Hin). injection Heq as <-.
      destruct (Hseconds v v' kind Hin) as [Hleaf _].
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne) Hleaf Hv').
    + intros v kind s Hin.
      apply in_pairs_lets in Hin as (v' & -> & Hin).
      destruct (Hseconds v v' kind Hin) as [_ (w' & Hl' & Hk')].
      exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k Hpath.
      destruct (Hpaths p v k Hpath) as (v' & w' & Hin & Hl' & Hpw).
      exists (Var v'), w'.
      split; [ exact (pairs_in_lets pairs v v' k Hin) | ].
      split; [ exact Hl' | exact Hpw ].
    + apply (run_answer (related M CH CS) _ _ _ _ (length (pairs_lets pairs))
               Hrun).
      * exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* The load of a field unboxed in turn. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) Value) eqn:Hk; [ | exact I ].
    destruct (read_eval S vars d f s r0 Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    pose proof (fields_hold_nested vars' fields_y nested f w0 r0
                  (proj2 (proj1 Hvars) y fields_y w0 Hy Hw0) Hin Hfield)
      as Hhold.
    destruct (pairs_hold vars' fields nested r0 pairs Hl Hhold)
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case_c vars vars' x fields r0 (pairs_lets pairs) body'
                conts' S' empty_heap rcd Hvars Hfields Hxc)
      as (vars1 & Hrun & Hrel).
    + rewrite pairs_lets_variables. exact Hfirsts.
    + intros v kind0 s v' Hin0 Hv' ->.
      apply in_pairs_lets in Hin0 as (v'' & Heq & Hin0).
      injection Heq as <-.
      destruct (Hseconds v v' kind0 Hin0) as [Hleaf _].
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne)
               (leaves_unboxed_in Value fields_y f (Unboxed nested) v' Hin
                  Hleaf)
               Hv').
    + intros v kind0 s Hin0.
      apply in_pairs_lets in Hin0 as (v' & -> & Hin0).
      destruct (Hseconds v v' kind0 Hin0) as [_ (w' & Hl' & Hk')].
      exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k0 Hpath.
      destruct (Hpaths p v k0 Hpath) as (v' & w' & Hin0 & Hl' & Hpw).
      exists (Var v'), w'.
      split; [ exact (pairs_in_lets pairs v v' k0 Hin0) | ].
      split; [ exact Hl' | exact Hpw ].
    + apply (run_answer (related M CH CS) _ _ _ _ (length (pairs_lets pairs))
               Hrun).
      * exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* A read of a field. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) kind) eqn:Hk; [ | exact I ].
    destruct (read_eval S vars d f s r0 Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    assert (Hp : upath Value (Unboxed fields) [f] v (field_kind f))
      by (econstructor; [ exact Hin | constructor ]).
    destruct (proj2 (proj1 Hvars) y fields w0 Hfields Hw0 [f] v _ Hp)
      as (w' & Hpw & Hv & _).
    cbn [path_value] in Hpw.
    destruct (field_value f w0) as [w1 | ] eqn:Hfw; [ | discriminate Hpw ].
    injection Hpw as <-.
    rewrite <- (Hfield w1 eq_refl) in Hv.
    right. left. eexists. split.
    + rewrite step_let_simple. cbn [eval_simple]. rewrite Hv, Hk.
      reflexivity.
    + exists D0; split; [ exact Hbody | ].
      split; [ exact (env_rel_c_bind _ _ x r0 Hvars Hx) | ].
      split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* A set whose representation changes. *)
    rewrite step_let_exec. cbn [exec_named eval_named].
    destruct (lookup (cr_current r) decls) as [code | ] eqn:Hcode;
      [ | exact I ].
    destruct (eval_value_slots vars vs) as [vs1 | ] eqn:Hvs1; [ | exact I ].
    destruct (new_slots_eval vars vars' r vs vs1 (proj1 Hvars) Hok Hnd Hvs1)
      as (vs2 & Hvs2 & Hslots).
    cbn. right. left. eexists. split.
    + rewrite step_let_exec. cbn [exec_named eval_named].
      rewrite Hdecls, Hvs2. reflexivity.
    + exists D0; split; [ exact Hbody | ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | repeat split ]]].
      apply (env_rel_c_bind_changed vars vars' x r _ _ Hvars Hxr).
      exists decls, vs1, vs2.
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ rewrite Hdecls, Hcode; reflexivity | exact Hslots ].
  - (* A read of a new slot. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (slot_read_not_heap d w s Hd) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) kind) eqn:Hk; [ | exact I ].
    destruct (slot_read_var S vars d w s r0 Hd Hr) as (v1 & Hv1).
    destruct Hs as [-> | (k & ->)];
      [ | rewrite (slot_read_const S vars d w _ Hd) in Hr; discriminate Hr ].
    cbn [eval_simple] in Hv1.
    destruct (proj2 Hvars y r v1 Hyr Hv1) as (v2 & Hv2 & Hc).
    destruct (slot_read_crel S S' vars d w (Var y) y r v1 v2 r0 Hd
                (or_introl eq_refl) Hv1 Hc Hr)
      as (vs' & Hholds & Hproj).
    pose proof (Hholds _ Hin) as Hsl. cbn [slot_holds] in Hsl.
    right. left. eexists. split.
    + rewrite step_let_exec, exec_named_proj,
        (eval_proj_from S' vars' _ sl y v2 Hv2), Hproj, Hsl, Hk.
      reflexivity.
    + exists D0; split; [ exact Hbody | ].
      split; [ exact (env_rel_c_bind _ _ x r0 Hvars Hx) | ].
      split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* The reads of the new slots of an unboxed value. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (slot_read_not_heap d w s Hd) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) Value) eqn:Hk; [ | exact I ].
    destruct (slot_read_var S vars d w s r0 Hd Hr) as (v1 & Hv1).
    destruct Hs as [-> | (k & ->)];
      [ | rewrite (slot_read_const S vars d w _ Hd) in Hr; discriminate Hr ].
    cbn [eval_simple] in Hv1.
    destruct (proj2 Hvars y r v1 Hyr Hv1) as (v2 & Hv2 & Hc).
    destruct (slot_read_crel S S' vars d w (Var y) y r v1 v2 r0 Hd
                (or_introl eq_refl) Hv1 Hc Hr)
      as (vs' & Hholds & Hproj).
    pose proof (Hholds _ Hin) as Hhold. cbn [slot_holds] in Hhold.
    destruct (pairs_hold vs' fields nested r0 pairs Hl Hhold)
      as (Hfirsts & Hseconds & Hpaths).
    destruct (proj_lets_case vars vars' x y fields r0 (current_slot r) pairs
                body' conts' S' empty_heap rcd Hvars Hfields Hxc Hfirsts)
      as (vars1 & Hrun & Hrel).
    + intros Hin'.
      pose proof (field_variables_in M x fields y Hfields Hin') as Hfv.
      apply among_In in Hfv. rewrite (CH_fresh y r Hyr) in Hfv.
      discriminate Hfv.
    + intros v s0 k Hin0.
      destruct (Hseconds v s0 k Hin0) as [_ (w' & Hl' & Hk')].
      exists w'. rewrite (eval_proj_from S' vars' _ s0 y v2 Hv2), Hproj.
      split; assumption.
    + intros p v k Hp.
      destruct (Hpaths p v k Hp) as (s0 & w' & Hin0 & Hl' & Hpw).
      exists s0, w'. split; [ exact Hin0 | ].
      rewrite (eval_proj_from S' vars' _ s0 y v2 Hv2), Hproj.
      split; assumption.
    + apply (run_answer (related M CH CS) _ _ _ _ (length pairs) Hrun).
      * exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* A [Let_static]. *)
    rewrite step_let_static.
    destruct (eval_static_bindings vars bs S) as [S1 | ] eqn:Hev;
      [ | exact I ].
    destruct (bindings_eval_rel bs bs' Hbs vars vars' S S' S1 Hvars Hst Hev)
      as (S1' & Hev' & Hst1).
    right. left. eexists.
    split; [ rewrite step_let_static, Hev'; reflexivity | ].
    exists D0; split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ exact Hconts | split; [ exact Hst1 | repeat split ] ].
  - (* A non-recursive [Let_cont]. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    exists (without [k] D0).
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    intros k0 Hk0. cbn [state_conts]. unfold resolve. rewrite !lookup_cons.
    destruct (Pos.eqb k0 k) eqn:E.
    + apply (cont_rel_closure M CH CS D0 ps md h h' vars vars' conts conts'
               [] [] Hps); cbn [map]; rewrite ?without_nil;
        [ exact Hh | exact Hvars | exact Hconts | constructor ].
    + apply Hconts. intros H. apply Hk0. apply in_without_iff.
      split; [ exact H | ]. intros [He | []]. subst k0.
      rewrite Pos.eqb_refl in E. discriminate E.
  - (* A recursive [Let_cont]. *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics hs) eqn:Hb; [ exact I | ].
    assert (Hb' : handlers_bind_statics hs' = false).
    { destruct (handlers_bind_statics hs') eqn:E; [ | reflexivity ].
      rewrite (proj2 (proj2 (proj1 (proj2 rr_shape) _ hs hs' Hhs)) E) in Hb.
      discriminate Hb. }
    right. left. eexists. split.
    { rewrite step_let_cont_recursive, Hb'. reflexivity. }
    exists (without (map fst hs) D0).
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ apply rec_closures_rel; assumption | ].
    split; [ exact Hst | repeat split ].
  - (* An [Apply_cont]. *)
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    pose proof (cont_answered _ _ vs S S' rcd
                  (resolve_rel D0 conts conts' (ac_continuation ac) Hconts
                     Hk)
                  Hst)
      as Ha.
    destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs S
                empty_heap rcd)
      as [s1 | k ws hp1 rd1 | ] eqn:Happ; [ | | exact I ].
    + destruct Ha as (t1 & Ht1 & Hrel). right. left. exists t1.
      split; [ | exact Hrel ].
      rewrite step_apply_cont,
        <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hac), Hvs.
      unfold apply_continuation. exact Ht1.
    + exists ws, rd1. split; [ | reflexivity ].
      rewrite step_apply_cont,
        <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hac), Hvs.
      unfold apply_continuation. exact Ha.
  - (* A [Switch]. *)
    rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    right. left. eexists. split.
    + rewrite step_switch,
        <- (plain_simple_eval M vars vars' scrutinee (proj1 Hvars)
              Hscrutinee),
        Hv, Hi, Harm.
      reflexivity.
    + exists D0. split; [ | split; [ exact Hvars | ] ].
      * destruct (find_arm_in i arms ac Harm) as (d & Hin).
        constructor; [ | exact (Hks _ Hin) ].
        rewrite forallb_forall in Harms. exact (Harms _ Hin).
      * split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* [Invalid]. *)
    rewrite step_invalid. exact I.
  - (* A call. *)
    pose proof (call_entered ap (ap_continuation ap) id ci vars vars' conts
                  conts' S S' rcd
                  (conj Hkind (conj Hci (conj Hcallee
                     (conj Hargs (conj Hextra Hlen)))))
                  Hvars Hst)
      as Hc.
    rewrite with_continuation_same in Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | | ]; [ | contradiction | exact I ].
    destruct Hc as (cc & cc' & hv & hv' & -> & Ht & Hrel & Hcbody).
    right. left. eexists. split; [ exact Ht | ].
    exists (code_dead ci cc).
    split; [ exact Hcbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    apply code_conts_rel; [ intros _ | ]; apply Hconts; assumption.
  - (* A call of code that never returns. *)
    pose proof (call_entered ap k id ci vars vars' conts conts' S S' rcd Hok
                  Hvars Hst)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | | ]; [ | contradiction | exact I ].
    destruct Hc as (cc & cc' & hv & hv' & -> & Ht & Hrel & Hcbody).
    right. left. eexists. split; [ exact Ht | ].
    exists (code_dead ci cc).
    split; [ exact Hcbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    apply code_conts_rel;
      [ intros H; rewrite Hnv in H; discriminate H | apply Hconts; exact Hke ].
  - (* The same, the second run returning to a handler [Invalid]. *)
    pose proof (call_entered ap w id ci vars vars' conts
                  ((w, Closure (Handler wps wmd Invalid) vars' conts' [])
                   :: conts')
                  S S' rcd Hok Hvars Hst)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | | ]; [ | contradiction | exact I ].
    destruct Hc as (cc & cc' & hv & hv' & -> & Ht & Hrel & Hcbody).
    right. right. right. eexists. exists 0. split.
    { exact (run_two _ _ _ (step_let_cont_non_recursive _ _ _ _ _ _ _ _) Ht). }
    exists (code_dead ci cc).
    split; [ exact Hcbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    rewrite (resolve_other (ap_exn_continuation ap) w)
      by (intros He; exact (Hw (eq_sym He))).
    apply code_conts_rel;
      [ intros H; rewrite Hnv in H; discriminate H | apply Hconts; exact Hke ].
  - (* The same, the first run returning to a wrapper. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    exists (wc :: w :: D0).
    split; [ | split; [ exact Hvars | ] ].
    { apply (rr_call_never M CH CS _ ap w id ci Hok Hnv);
        [ right; left; reflexivity | ].
      intros [He | [He | He]];
        [ exact (Hwc He) | exact (Hw He) | exact (Hke He) ]. }
    split; [ | split; [ exact Hst | repeat split ] ].
    intros k0 Hk0. cbn [state_conts].
    rewrite (resolve_other k0 wc), (resolve_other k0 w);
      [ | intros ->; apply Hk0; right; left; reflexivity
        | intros ->; apply Hk0; left; reflexivity ].
    apply Hconts. intros H. apply Hk0. right. right. exact H.
  - (* A handler that only the first run binds. *)
    rewrite step_let_cont_non_recursive. right. right. left.
    split; [ unfold measure; simpl; lia | ].
    exists (k :: D0).
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    intros k0 Hk0. cbn [state_conts].
    rewrite (resolve_other k0 k) by (intros ->; apply Hk0; left; reflexivity).
    apply Hconts. intros H. apply Hk0. right. exact H.
Qed.

(* Related terms, run on their own, refine each other. *)
Lemma rr_refines (p q : expr) :
  rr M CH CS [] p q ->
  refines (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hrr. apply (shrinking_refines (related M CH CS) eq measure).
  - exact related_answered.
  - intros vs vs' ->. reflexivity.
  - exists []. split; [ exact Hrr | split; [ exact env_rel_c_empty | ] ].
    split; [ intros k _; constructor | ].
    split; [ | repeat split ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros id c Hc. discriminate Hc.
Qed.

End Simulation.

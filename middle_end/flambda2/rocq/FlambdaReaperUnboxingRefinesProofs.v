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

(* The proofs of FlambdaReaperUnboxingRefinesSpec.v. Nothing here needs
   review.

   The rebuild with unboxing is compared with the plain rebuild, which
   FlambdaReaperRebuildProofs.v compares with the unit. The two rebuilt terms
   are related by [ub]: equal but at the [Let] of an unboxed variable, which the
   unboxing makes [Let]s of the variables of its fields ([lets]: of what the
   allocation holds, of the variables of the fields of the variable it is an
   alias of, or, for a field unboxed in turn, of those of the variable that the
   allocation holds there: [alloc_entry]), each to the variable at the same
   path of fields of the other ([unboxed_field_pairs], [pairs_spec]), at any
   depth, at a read of a field of it, which it makes the variable of the
   field, at a read of a field unboxed in turn into an unboxed variable
   ([ub_load]), at a handler, whose unboxed parameters it makes the variables
   of their fields ([params_rel]), and at an [Apply_cont] to such a handler,
   whose unboxed arguments it makes the variables of their fields that the
   parameter has ([args_rel]). [ub] is indexed by the shapes of the
   parameters of the continuations in scope (which are unboxed, with which
   fields); those of continuations not free in a term do not matter
   ([ub_agree]). Related terms run alike ([ub_refines]), by a simulation of
   FlambdaReaperRebuildProofs.v ([shrinking_refines]): a step of the plain term
   is answered by as many steps as the unboxing makes [Let]s, none where it
   makes none; an unboxed variable bound in the plain run to a value has the
   variables of its fields bound in the other to the values at their paths of
   fields in the value ([upath], [path_value], [fields_hold], [env_rel]), so
   that the arguments of an [Apply_cont] related are bound to
   parameters related ([bind_args_rel]). Immutable blocks and boxed numbers are
   values, so no heap is involved.

   The two rebuilds of a unit of the fragment are related ([rebuild_ub]),
   the plain rebuild with the masks of the handlers in scope and the
   rebuild with unboxing with their decisions, which the fragment's view
   of them ([conts_match]) relates. What the plain rebuild keeps of an
   allocation that is unboxed, it keeps because the solution is closed
   ([field_kept]): each field unboxed is read with a use from a use of
   the variable ([get_fields_in] and the lemmas of "Where the unboxed
   fields come from"), the graph of the unit reads no function slot and
   makes no call ([closure_free_reads_blocks]) so that each such use is a
   use of the variable, and the facts of the allocation are those of
   FlambdaReaperFlowProofs.expr_edges. A read that the rebuild keeps of
   an unboxed variable finds its field ([reads_found]), by
   [check_unboxed_reads]: the variable read into has a use, so that the
   field is among those read from the uses of the variable
   ([get_fields_found]; the variable has no [any_usage], so that it is
   one of its uses), the field of the allocation has a source, and the
   variable read into is not to be unboxed, so that the field is a
   variable ([mk_unboxed_fields_lookup], [unboxed_fields_of_names_dom],
   [unboxed_fields_of_names_bound]). The variables of the fields are
   fresh ([unit_unboxed_fields_fresh]), so that the checks of the
   fragment hold with "plain" read as "neither unboxed nor the variable
   of a field" ([strong_local]), which is what the relation asks. The
   statements compose. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaRenameApartSpec FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaFreshQuantifierSpec
  FlambdaPrimitiveProofs FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* The fragment, with the names of the fields fresh                         *)
(* ======================================================================== *)

(* The fragment of FlambdaReaperUnboxingRefinesSpec.v asks that unboxed
   variables be only read; the proofs use too that the variables of the
   fields are no variable of the unit, which the model makes them
   ([unit_unboxed_fields_fresh], below): the same checks, a variable or a
   simple being [plain] if it is not unboxed and not the variable of a
   field. *)
Section Fragment.

Variable unboxed : list (node * unboxed_fields).
Variable sol : solution.

(* The variables that stand for the unboxed fields. *)
Definition field_variables : list variable :=
  flat_map (fun '(_, fields) => map fst (fields_leaves fields)) unboxed.

Definition plain_variable (x : variable) : bool :=
  negb (Keyed.is_bound node_eqb (Node_variable x) unboxed)
  && negb (among x field_variables).

Definition plain_simple (s : simple) : bool :=
  match s with
  | Var x => plain_variable x
  | Symbol sym => negb (Keyed.is_bound node_eqb (Node_symbol sym) unboxed)
  | Const _ => true
  end.

Definition unboxed_binding_ok (kinds : finite_map flambda_kind)
    (x : variable) (kind : flambda_kind) (d : named) (fields : unboxed_fields)
    : bool :=
  match kind, d with
  | Value, Prim (Variadic (Make_block (Block_kind_values _))) args =>
    forallb
      (fun '(f, u) =>
         match f, u with
         | Field_block i Value, Not_unboxed _ =>
           match nth_error args i with
           | Some arg =>
             plain_simple arg
             && kind_eqb (simple_kind (kinds_env kinds) arg) Value
           | None => false
           end
         | Field_block i Value, Unboxed nested =>
           match nth_error args i with
           | Some (Var b) =>
             negb (Pos.eqb x b)
             && kind_eqb (simple_kind (kinds_env kinds) (Var b)) Value
             && unboxed_arg_ok unboxed sol kinds true nested (Var b)
           | _ => false
           end
         | _, _ => false
         end)
      fields
  | Value, Prim (Unary (Box_number bn)) [contents] =>
    plain_simple contents
    && forallb
         (fun '(f, u) =>
            match u with
            | Not_unboxed _ => field_eqb f (Field_boxed_number bn)
            | Unboxed _ => false
            end)
         fields
  | Value, Simple (Var y) =>
    negb (Pos.eqb x y) && unboxed_arg_ok unboxed sol kinds true fields (Var y)
  | Value, Prim (Unary (Block_load _ _ i)) [Var y] =>
    negb (Pos.eqb x y)
    && match fields_of unboxed y with
       | Some fields_y =>
         match Keyed.lookup field_eqb (Field_block (Z.to_nat i) Value) fields_y
         with
         | Some (Unboxed nested) => fields_alike fields nested
         | _ => false
         end
       | None => false
       end
  | _, _ => false
  end.

Definition plain_binding_ok (x : variable) (kind : flambda_kind) (d : named)
    : bool :=
  match unboxed_read unboxed d with
  | Some _ => read_shape_ok kind d
  | None =>
    match d with
    | Simple s => plain_simple s
    | Prim _ args => forallb plain_simple args
    | _ => false
    end
  end.

Definition local_binding (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb plain_simple fields
  | _ => false
  end.

Fixpoint args_ok (kinds : finite_map flambda_kind)
    (params : list (variable * bool)) (args : list simple) : bool :=
  match params, args with
  | _, [] => true
  | [], _ :: _ => forallb plain_simple args
  | (p, keep) :: params, a :: args =>
    match fields_of unboxed p with
    | Some fields => unboxed_arg_ok unboxed sol kinds keep fields a
    | None => negb keep || plain_simple a
    end
    && args_ok kinds params args
  end.

Definition apply_cont_ok (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (ac : apply_cont) : bool :=
  match lookup (ac_continuation ac) conts with
  | Some params => args_ok kinds params (ac_args ac)
  | None => forallb plain_simple (ac_args ac)
  end.

Definition param_ok (p : variable) (kind : flambda_kind) (keep : bool)
    : bool :=
  negb (among p field_variables)
  && FlambdaReaperUnboxingRefinesSpec.param_ok unboxed p kind keep.

Fixpoint local_expr (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (e : expr) : bool :=
  match e with
  | Let x kind d body =>
    negb (among x field_variables)
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok kinds x kind d fields
       | None => plain_binding_ok x kind d
       end
    && local_expr ((x, kind) :: kinds) conts body
  | Let_static bindings body =>
    forallb local_binding bindings && local_expr kinds conts body
  | Let_cont_non_recursive k h body =>
    let mask := handler_mask sol (handler_is_exn_handler h) h in
    local_handler kinds conts mask h
    && local_expr kinds ((k, combine (handler_variables h) mask) :: conts)
         body
  | Let_cont_recursive handlers body =>
    let conts :=
      map
        (fun '(k, h) =>
           (k, combine (handler_variables h) (handler_mask sol false h)))
        handlers
      ++ conts
    in
    forallb
      (fun '(_, h) => local_handler kinds conts (handler_mask sol false h) h)
      handlers
    && local_expr kinds conts body
  | Apply _ => false
  | Apply_cont ac => apply_cont_ok kinds conts ac
  | Switch scrutinee arms =>
    plain_simple scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok kinds conts ac) arms
  | Invalid => true
  end
with local_handler (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (mask : list bool)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb (fun '((p, kind), keep) => param_ok p kind keep)
         (combine (combine params (hm_params_kinds metadata)) mask)
    && local_expr (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         conts handler
  end.

End Fragment.

(* ======================================================================== *)
(* The paths to the variables of unboxed fields, and their pairs           *)
(* ======================================================================== *)

(* [upath k u p v k']: the variable [v] of the unboxed field [u], of kind
   [k], is at the path of fields [p], and of kind [k']. *)
Inductive upath : flambda_kind -> unboxed_field -> list field -> variable
    -> flambda_kind -> Prop :=
  | upath_leaf (k : flambda_kind) (v : variable) :
    upath k (Not_unboxed v) [] v k
  | upath_field (k : flambda_kind) (fields : unboxed_fields) (f : field)
      (u : unboxed_field) (p : list field) (v : variable)
      (k' : flambda_kind) :
    (f, u) ∈ fields -> upath (field_kind f) u p v k' ->
    upath k (Unboxed fields) (f :: p) v k'.

(* The [Let]s of the variables of [unboxed_field_pairs], each to the
   other. *)
Definition pairs_lets (pairs : list (variable * variable * flambda_kind))
    : list (variable * flambda_kind * simple) :=
  map (fun '(x, x', kind) => (x, kind, Var x')) pairs.

Section Unboxed_field_ind.

Variable P : unboxed_field -> Prop.
Hypothesis P_not_unboxed : ∀ v : variable, P (Not_unboxed v).
Hypothesis P_unboxed : ∀ fields : unboxed_fields,
  (∀ (f : field) (u : unboxed_field), (f, u) ∈ fields -> P u) ->
  P (Unboxed fields).

Fixpoint unboxed_field_rec (u : unboxed_field) : P u :=
  match u with
  | Not_unboxed v => P_not_unboxed v
  | Unboxed fields =>
    P_unboxed fields
      ((fix go (l : list (field * unboxed_field))
           : ∀ (f : field) (u : unboxed_field), (f, u) ∈ l -> P u :=
          match l with
          | [] => fun f u H => match H with end
          | (f0, u0) :: l =>
            fun f u H =>
              match H with
              | or_introl Heq =>
                match Heq in _ = fu return P (snd fu) with
                | eq_refl => unboxed_field_rec u0
                end
              | or_intror H' => go l f u H'
              end
          end) fields)
  end.

End Unboxed_field_ind.

Lemma upath_unboxed_kind (k k2 : flambda_kind) (fields : unboxed_fields)
    (p : list field) (v : variable) (k' : flambda_kind) :
  upath k (Unboxed fields) p v k' -> upath k2 (Unboxed fields) p v k'.
Proof.
  intros H. inversion H; subst. econstructor; eassumption.
Qed.

Lemma upath_cons (k : flambda_kind) (fu : field * unboxed_field)
    (fields : unboxed_fields) (p : list field) (v : variable)
    (k' : flambda_kind) :
  upath k (Unboxed fields) p v k' -> upath k (Unboxed (fu :: fields)) p v k'.
Proof.
  intros H. inversion H; subst. econstructor; [ right; eassumption | ].
  eassumption.
Qed.

Lemma upath_unboxed_inv (k : flambda_kind) (fields : unboxed_fields)
    (p : list field) (v : variable) (k' : flambda_kind) :
  upath k (Unboxed fields) p v k' ->
  ∃ (f : field) (u : unboxed_field) (p0 : list field),
    p = f :: p0 ∧ (f, u) ∈ fields ∧ upath (field_kind f) u p0 v k'.
Proof. intros H. inversion H; subst. eexists _, _, _. eauto. Qed.

Lemma upath_not_unboxed_inv (k : flambda_kind) (x : variable)
    (p : list field) (v : variable) (k' : flambda_kind) :
  upath k (Not_unboxed x) p v k' -> p = [] ∧ v = x ∧ k' = k.
Proof. intros H. inversion H; subst. auto. Qed.

Lemma field_eqb_decides : decides_equality field_eqb.
Proof.
  intros f f'. split; [ apply field_eqb_eq | intros <-; apply field_eqb_refl ].
Qed.

(* What [unboxed_field_pairs] pairs: the variables at the same path of
   the two, of the kind at that path; each variable of the first, in
   order. *)
Lemma pairs_spec (u : unboxed_field) :
  ∀ (k : flambda_kind) (u' : unboxed_field)
    (pairs : list (variable * variable * flambda_kind)),
    unboxed_field_pairs k u u' = Some pairs ->
    (∀ (x x' : variable) (k' : flambda_kind),
       (x, x', k') ∈ pairs ->
       ∃ p : list field, upath k u p x k' ∧ upath k u' p x' k')
    ∧ (∀ (p : list field) (v : variable) (k' : flambda_kind),
         upath k u p v k' ->
         ∃ v' : variable, (v, v', k') ∈ pairs ∧ upath k u' p v' k')
    ∧ map (fun '(x, _, _) => x) pairs = map fst (unboxed_field_leaves k u)
    ∧ map (fun '(_, _, k') => k') pairs
      = map snd (unboxed_field_leaves k u).
Proof.
  induction u as [v | fields IH] using unboxed_field_rec;
    intros k u' pairs Hp.
  - destruct u' as [v' | fields']; [ | discriminate Hp ].
    injection Hp as <-. split; [ | split; [ | split; reflexivity ] ].
    + intros x x' k' [Heq | []]. injection Heq as <- <- <-.
      exists []. split; constructor.
    + intros p v0 k' Hpath. apply upath_not_unboxed_inv in Hpath
        as (-> & -> & ->).
      exists v'. split; [ left; reflexivity | constructor ].
  - destruct u' as [v' | fields']; [ discriminate Hp | ].
    revert pairs Hp. induction fields as [ | [f0 u0] rest IHrest];
      intros pairs Hp.
    + injection Hp as <-. split; [ intros x x' k' [] | ].
      split; [ | split; reflexivity ].
      intros p v k' Hpath. apply upath_unboxed_inv in Hpath
        as (f & u & p0 & _ & [] & _).
    + change (unboxed_field_pairs k (Unboxed ((f0, u0) :: rest))
                (Unboxed fields'))
        with (match Keyed.lookup field_eqb f0 fields' with
              | Some u' =>
                match unboxed_field_pairs (field_kind f0) u0 u',
                      unboxed_field_pairs k (Unboxed rest) (Unboxed fields')
                with
                | Some p, Some q => Some (p ++ q)
                | _, _ => None
                end
              | None => None
              end) in Hp.
      destruct (Keyed.lookup field_eqb f0 fields') as [u0' | ] eqn:Hl;
        [ | discriminate Hp ].
      pose proof (keyed_lookup_in field_eqb field_eqb_decides f0 _ fields'
                    Hl) as Hin0.
      destruct (unboxed_field_pairs (field_kind f0) u0 u0') as [p1 | ]
        eqn:Hp1; [ | discriminate Hp ].
      destruct (unboxed_field_pairs k (Unboxed rest) (Unboxed fields'))
        as [q | ] eqn:Hq; [ | discriminate Hp ].
      injection Hp as <-.
      destruct (IH f0 u0 (or_introl eq_refl) (field_kind f0) u0' p1 Hp1)
        as (Ha1 & Hb1 & Hc1 & Hd1).
      destruct (IHrest (fun f u Hin => IH f u (or_intror Hin)) q eq_refl)
        as (Ha2 & Hb2 & Hc2 & Hd2).
      split; [ | split; [ | split ] ].
      * intros x x' k' Hin. apply in_app_or in Hin as [Hin | Hin].
        -- destruct (Ha1 x x' k' Hin) as (p & H1 & H2). exists (f0 :: p).
           split; econstructor; try eassumption; left; reflexivity.
        -- destruct (Ha2 x x' k' Hin) as (p & H1 & H2). exists p.
           split; [ apply upath_cons; exact H1 | exact H2 ].
      * intros p v k' Hpath.
        apply upath_unboxed_inv in Hpath as (f & u & p0 & -> & Hin & Hp0).
        destruct Hin as [Heq | Hin].
        -- injection Heq as <- <-.
           destruct (Hb1 p0 v k' Hp0) as (v' & Hv' & Hp0').
           exists v'. split; [ apply in_or_app; left; exact Hv' | ].
           econstructor; eassumption.
        -- assert (Hr : upath k (Unboxed rest) (f :: p0) v k')
             by (econstructor; eassumption).
           destruct (Hb2 _ v k' Hr) as (v' & Hv' & Hp').
           exists v'. split; [ apply in_or_app; right; exact Hv' | ].
           exact Hp'.
      * rewrite map_app, Hc1, Hc2. simpl. rewrite map_app. reflexivity.
      * rewrite map_app, Hd1, Hd2. simpl. rewrite map_app. reflexivity.
Qed.

(* ======================================================================== *)
(* Terms related by local unboxing                                          *)
(* ======================================================================== *)

Section Related.

(* The unboxed fields of the names; their variables are distinct. *)
Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

(* [Let]s of simples, the first the innermost. *)
Definition lets (l : list (variable * flambda_kind * simple)) (body : expr)
    : expr :=
  List.fold_left (fun body '(v, kind, s) => Let v kind (Simple s) body) l body.

(* The variable of the field [f] of the allocation [d] is bound to [s], at
   [kind]. *)
Definition alloc_field (d : named) (f : field) (kind : flambda_kind)
    (s : simple) : Prop :=
  match d with
  | Prim (Variadic (Make_block (Block_kind_values _))) args =>
    ∃ i : nat,
      f = Field_block i Value ∧ kind = Value ∧ nth_error args i = Some s
  | Prim (Unary (Box_number bn)) [c] =>
    f = Field_boxed_number bn ∧ kind = unboxed_kind bn ∧ s = c
  | _ => False
  end.

(* The field [f] of a value. *)
Definition field_value (f : field) (w : value) : option value :=
  match f, w with
  | Field_block i _, Value_block _ vs => nth_error vs i
  | Field_boxed_number _, Value_boxed_number c => Some (Value_const c)
  | _, _ => None
  end.

(* The value at a path of fields. *)
Fixpoint path_value (p : list field) (w : value) : option value :=
  match p with
  | [] => Some w
  | f :: p =>
    match field_value f w with
    | Some w1 => path_value p w1
    | None => None
    end
  end.

(* The read of a field: [Block_load] or [Unbox_number]. *)
Definition read_of (d : named) : option (field * simple) :=
  match d with
  | Prim (Unary (Block_load _ _ _)) [_] | Prim (Unary (Unbox_number _)) [_] =>
    field_read d
  | _ => None
  end.

Definition named_plain (d : named) : bool :=
  match d with
  | Simple s => plain_simple M s
  | Prim _ args => forallb (plain_simple M) args
  | _ => false
  end.

(* What the unboxing binds for the field [f] of the allocation [d] of
   [x]: the variable of the field to what [d] holds there, a simple of
   [plain_simple]; or, where the field is unboxed in turn, the variables
   of its fields to those of the variable that [d] holds there, another
   unboxed variable. *)
Definition alloc_entry (x : variable) (d : named)
    (fu : field * unboxed_field)
    (l : list (variable * flambda_kind * simple)) : Prop :=
  match fu with
  | (f, Not_unboxed v) =>
    ∃ (kind : flambda_kind) (s : simple),
      l = [(v, kind, s)] ∧ alloc_field d f kind s ∧ plain_simple M s = true
  | (f, Unboxed nested) =>
    ∃ (b : variable) (fields_b : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)),
      alloc_field d f Value (Var b) ∧ fields_of M b = Some fields_b ∧ x <> b
      ∧ unboxed_field_pairs Value (Unboxed nested) (Unboxed fields_b)
        = Some pairs
      ∧ l = pairs_lets pairs
  end.

(* The parameters of a handler as each run has them, of the shapes [sh]:
   one not unboxed ([None]) is a parameter of both; one unboxed with the
   fields [fields] ([Some fields]), of kind [Value], is in the second the
   variables of its fields. *)
Inductive params_rel : list (option unboxed_fields)
    -> list (variable * flambda_kind) -> list (variable * flambda_kind)
    -> Prop :=
  | params_rel_nil : params_rel [] [] []
  | params_rel_plain (sh : list (option unboxed_fields)) (p : variable)
      (kind : flambda_kind) (ps ps' : list (variable * flambda_kind)) :
    plain_variable M p = true -> params_rel sh ps ps' ->
    params_rel (None :: sh) ((p, kind) :: ps) ((p, kind) :: ps')
  | params_rel_unboxed (sh : list (option unboxed_fields)) (p : variable)
      (fields : unboxed_fields) (ps ps' : list (variable * flambda_kind)) :
    fields_of M p = Some fields -> among p (field_variables M) = false ->
    params_rel sh ps ps' ->
    params_rel (Some fields :: sh) ((p, Value) :: ps)
      (fields_leaves fields ++ ps').

(* The arguments given to a handler of the shapes [sh]: a simple of
   [plain_simple] in both; for an unboxed parameter, an unboxed variable
   in the first and, in the second, the variables of its fields that the
   parameter has. Missing arguments are missing in both, and those beyond
   the parameters are alike. *)
Inductive args_rel : list (option unboxed_fields) -> list simple
    -> list simple -> Prop :=
  | args_rel_beyond (args : list simple) :
    forallb (plain_simple M) args = true -> args_rel [] args args
  | args_rel_missing (o : option unboxed_fields)
      (sh : list (option unboxed_fields)) :
    args_rel (o :: sh) [] []
  | args_rel_plain (sh : list (option unboxed_fields)) (a : simple)
      (args args' : list simple) :
    plain_simple M a = true -> args_rel sh args args' ->
    args_rel (None :: sh) (a :: args) (a :: args')
  | args_rel_unboxed (sh : list (option unboxed_fields))
      (fields fields_y : unboxed_fields) (y : variable)
      (pairs : list (variable * variable * flambda_kind))
      (args args' : list simple) :
    fields_of M y = Some fields_y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs ->
    args_rel sh args args' ->
    args_rel (Some fields :: sh) (Var y :: args)
      (map (fun '(_, x', _) => Var x') pairs ++ args').

(* The shapes of the parameters of a continuation; one that is not bound
   is a way out, whose arguments are alike. *)
Definition shapes_at (K : finite_map (list (option unboxed_fields)))
    (k : continuation) : list (option unboxed_fields) :=
  match lookup k K with
  | Some sh => sh
  | None => []
  end.

Definition ac_rel (K : finite_map (list (option unboxed_fields)))
    (ac ac' : apply_cont) : Prop :=
  ac_continuation ac' = ac_continuation ac
  ∧ ac_trap_action ac' = ac_trap_action ac
  ∧ args_rel (shapes_at K (ac_continuation ac)) (ac_args ac) (ac_args ac').

(* [K] are the shapes of the parameters of the continuations in scope. *)
Inductive ub : finite_map (list (option unboxed_fields)) -> expr -> expr
    -> Prop :=
  | ub_let (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (body body' : expr) :
    plain_variable M x = true -> named_plain d = true ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    ub K body body' ->
    ub K (Let x kind d body) (Let x kind d body')
  | ub_alloc (K : finite_map (list (option unboxed_fields))) (x : variable)
      (d : named) (fields : unboxed_fields)
      (ls : list (list (variable * flambda_kind * simple)))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    Forall2 (alloc_entry x d) fields ls ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    ub K body body' ->
    ub K (Let x Value d body) (lets (concat ls) body')
  | ub_alias (K : finite_map (list (option unboxed_fields))) (x y : variable)
      (fields fields_y : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    fields_of M y = Some fields_y -> x <> y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs ->
    ub K body body' ->
    ub K (Let x Value (Simple (Var y)) body) (lets (pairs_lets pairs) body')
  | ub_load (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (f : field) (s : simple)
      (y : variable) (fields fields_y nested : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields_y -> x <> y ->
    (f, Unboxed nested) ∈ fields_y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed nested)
    = Some pairs ->
    ub K body body' ->
    ub K (Let x Value d body) (lets (pairs_lets pairs) body')
  | ub_read (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (f : field)
      (s : simple) (y v : variable) (fields : unboxed_fields)
      (body body' : expr) :
    plain_variable M x = true ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields ->
    (f, Not_unboxed v) ∈ fields ->
    ub K body body' ->
    ub K (Let x kind d body) (Let x kind (Simple (Var v)) body')
  | ub_static (K : finite_map (list (option unboxed_fields)))
      (bindings : list static_binding) (body body' : expr) :
    forallb (local_binding M) bindings = true -> ub K body body' ->
    ub K (Let_static bindings body) (Let_static bindings body')
  | ub_let_cont (K : finite_map (list (option unboxed_fields)))
      (k : continuation) (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata)
      (handler handler' body body' : expr) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ub K handler handler' ->
    ub ((k, sh) :: K) body body' ->
    ub K
      (Let_cont_non_recursive k (Handler (map fst ps) metadata handler) body)
      (Let_cont_non_recursive k (Handler (map fst ps') metadata' handler')
         body')
  | ub_let_cont_rec (K S : finite_map (list (option unboxed_fields)))
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    ub_handlers (S ++ K) handlers handlers' S ->
    handlers_bind_statics handlers = handlers_bind_statics handlers' ->
    ub (S ++ K) body body' ->
    ub K (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | ub_apply_cont (K : finite_map (list (option unboxed_fields)))
      (ac ac' : apply_cont) :
    ac_rel K ac ac' -> ub K (Apply_cont ac) (Apply_cont ac')
  | ub_switch (K : finite_map (list (option unboxed_fields)))
      (scrutinee : simple) (arms arms' : list (Z * apply_cont)) :
    plain_simple M scrutinee = true ->
    Forall2
      (fun (arm arm' : Z * apply_cont) =>
         fst arm' = fst arm ∧ ac_rel K (snd arm) (snd arm'))
      arms arms' ->
    ub K (Switch scrutinee arms) (Switch scrutinee arms')
  | ub_invalid (K : finite_map (list (option unboxed_fields))) :
    ub K Invalid Invalid
with ub_handlers : finite_map (list (option unboxed_fields))
    -> list (continuation * continuation_handler)
    -> list (continuation * continuation_handler)
    -> finite_map (list (option unboxed_fields)) -> Prop :=
  | ub_handlers_nil (K : finite_map (list (option unboxed_fields))) :
    ub_handlers K [] [] []
  | ub_handlers_cons (K : finite_map (list (option unboxed_fields)))
      (k : continuation) (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (handlers handlers' : list (continuation * continuation_handler))
      (S : finite_map (list (option unboxed_fields))) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ub K handler handler' ->
    ub_handlers K handlers handlers' S ->
    ub_handlers K ((k, Handler (map fst ps) metadata handler) :: handlers)
      ((k, Handler (map fst ps') metadata' handler') :: handlers')
      ((k, sh) :: S).

(* The variables of the fields [fields] of a value [w], at any depth,
   bound in [vars'] to the values at their paths, of their kinds. *)
Definition fields_hold (vars' : env) (fields : unboxed_fields) (w : value)
    : Prop :=
  ∀ (p : list field) (v : variable) (k : flambda_kind),
    upath Value (Unboxed fields) p v k ->
    ∃ w' : value,
      path_value p w = Some w' ∧ lookup v vars' = Some w'
      ∧ kind_of_value w' = k.

(* The environments: a variable of neither kind has one value in both;
   an unboxed variable bound in the first has the variables of its
   fields bound in the second to the values at their paths in its
   value. *)
Definition env_rel (vars vars' : env) : Prop :=
  (∀ z : variable, plain_variable M z = true -> lookup z vars = lookup z vars')
  ∧ ∀ (x : variable) (fields : unboxed_fields) (w : value),
      fields_of M x = Some fields -> lookup x vars = Some w ->
      fields_hold vars' fields w.

Inductive cont_rel : list (option unboxed_fields) -> cont_value -> cont_value
    -> Prop :=
  | cont_rel_closure (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (vars vars' : env) (K S : finite_map (list (option unboxed_fields)))
      (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ub (S ++ K) handler handler' ->
    env_rel vars vars' -> conts_rel K conts conts' ->
    ub_handlers (S ++ K) recursive recursive' S ->
    cont_rel sh
      (Closure (Handler (map fst ps) metadata handler) vars conts recursive)
      (Closure (Handler (map fst ps') metadata' handler') vars' conts'
         recursive')
with conts_rel : finite_map (list (option unboxed_fields)) -> cont_env
    -> cont_env -> Prop :=
  | conts_rel_nil : conts_rel [] [] []
  | conts_rel_cons (k : continuation) (sh : list (option unboxed_fields))
      (c c' : cont_value) (K : finite_map (list (option unboxed_fields)))
      (conts conts' : cont_env) :
    cont_rel sh c c' -> conts_rel K conts conts' ->
    conts_rel ((k, sh) :: K) ((k, c) :: conts) ((k, c') :: conts').

Definition related (s t : state) : Prop :=
  ∃ K : finite_map (list (option unboxed_fields)),
    ub K (state_expr s) (state_expr t) ∧ env_rel (state_vars s) (state_vars t)
    ∧ conts_rel K (state_conts s) (state_conts t)
    ∧ state_statics s = state_statics t ∧ state_heap s = state_heap t
    ∧ state_reads s = state_reads t ∧ state_heap s = empty_heap.

End Related.

(* ======================================================================== *)
(* The variables of the fields                                              *)
(* ======================================================================== *)

Lemma node_eqb_decides : decides_equality node_eqb.
Proof.
  intros n n'. split; [ apply node_eqb_eq | intros <-; apply node_eqb_refl ].
Qed.

Section Fields.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Definition leaf_variables (fields : unboxed_fields) : list variable :=
  map fst (fields_leaves fields).

Lemma leaves_cons (f : field) (u : unboxed_field) (fields : unboxed_fields) :
  fields_leaves ((f, u) :: fields)
  = unboxed_field_leaves (field_kind f) u ++ fields_leaves fields.
Proof. reflexivity. Qed.

Lemma leaf_variables_nested (f : field) (nested rest : unboxed_fields) :
  leaf_variables ((f, Unboxed nested) :: rest)
  = leaf_variables nested ++ leaf_variables rest.
Proof. unfold leaf_variables. rewrite leaves_cons, map_app. reflexivity. Qed.

Lemma leaf_in (fields : unboxed_fields) (f : field) (v : variable) :
  (f, Not_unboxed v) ∈ fields -> v ∈ leaf_variables fields.
Proof.
  induction fields as [ | [f' u] fields IH]; intros Hin; [ destruct Hin | ].
  unfold leaf_variables. rewrite leaves_cons, map_app. apply in_or_app.
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. left. left. reflexivity.
  - right. exact (IH Hin).
Qed.

Lemma fields_of_in (x : variable) (fields : unboxed_fields) :
  fields_of M x = Some fields -> (Node_variable x, fields) ∈ M.
Proof. apply (keyed_lookup_in node_eqb node_eqb_decides). Qed.

Lemma field_variables_in (x : variable) (fields : unboxed_fields)
    (v : variable) :
  fields_of M x = Some fields -> v ∈ leaf_variables fields ->
  v ∈ field_variables M.
Proof.
  intros Hx Hv. apply fields_of_in in Hx. unfold field_variables.
  apply in_flat_map. exists (Node_variable x, fields). split; assumption.
Qed.

Lemma plain_variable_iff (z : variable) :
  plain_variable M z = true
  <-> Keyed.is_bound node_eqb (Node_variable z) M = false
      ∧ among z (field_variables M) = false.
Proof.
  unfold plain_variable. rewrite andb_true_iff, !negb_true_iff. reflexivity.
Qed.

Lemma plain_not_unboxed (z : variable) (fields : unboxed_fields) :
  plain_variable M z = true -> fields_of M z = Some fields -> False.
Proof.
  intros Hz Hf. apply plain_variable_iff in Hz as [Hz _].
  unfold Keyed.is_bound in Hz. unfold fields_of in Hf. rewrite Hf in Hz.
  discriminate.
Qed.

Lemma leaves_unboxed_in (k : flambda_kind) (fields : unboxed_fields)
    (f : field) (u : unboxed_field) (v : variable) :
  (f, u) ∈ fields -> v ∈ map fst (unboxed_field_leaves (field_kind f) u) ->
  v ∈ map fst (unboxed_field_leaves k (Unboxed fields)).
Proof.
  induction fields as [ | [f0 u0] fields IH]; intros Hin Hv;
    [ destruct Hin | ].
  change (unboxed_field_leaves k (Unboxed ((f0, u0) :: fields)))
    with (unboxed_field_leaves (field_kind f0) u0
          ++ unboxed_field_leaves k (Unboxed fields)).
  rewrite map_app. apply in_or_app. destruct Hin as [Heq | Hin];
    [ injection Heq as -> ->; left; exact Hv | right; exact (IH Hin Hv) ].
Qed.

(* The variable at a path is a variable of the fields. *)
Lemma upath_in_leaves (k : flambda_kind) (u : unboxed_field) (p : list field)
    (v : variable) (k' : flambda_kind) :
  upath k u p v k' -> v ∈ map fst (unboxed_field_leaves k u).
Proof.
  induction 1 as [k v | k fields f u p v k' Hin _ IH];
    [ left; reflexivity | ].
  exact (leaves_unboxed_in k fields f u v Hin IH).
Qed.

Lemma upath_leaf_in (fields : unboxed_fields) (p : list field)
    (v : variable) (k : flambda_kind) :
  upath Value (Unboxed fields) p v k -> v ∈ leaf_variables fields.
Proof. apply upath_in_leaves. Qed.

Lemma nodup_app_apart {A : Type} (l l' : list A) (a : A) :
  NoDup (l ++ l') -> a ∈ l -> a ∈ l' -> False.
Proof.
  induction l as [ | b l IH]; intros Hnd Ha Ha'; [ destruct Ha | ].
  simpl in Hnd. inversion Hnd as [ | b' l0 Hb Hnd' [Heq Heq']]; subst.
  destruct Ha as [<- | Ha].
  - apply Hb. apply in_or_app. right. exact Ha'.
  - exact (IH Hnd' Ha Ha').
Qed.

(* The variables of two entries of a list without repetition are
   apart. *)
Lemma flat_map_nodup_apart {A : Type} (g : A -> list positive)
    (l : list A) (a b : A) :
  NoDup (flat_map g l) -> a ∈ l -> b ∈ l -> a <> b ->
  ∀ v : positive, v ∈ g a -> v ∈ g b -> False.
Proof.
  induction l as [ | c l IH]; intros Hnd Ha Hb Hne v Hva Hvb; [ destruct Ha | ].
  simpl in Hnd. apply NoDup_app_remove_l in Hnd as Hrest.
  destruct Ha as [<- | Ha]; destruct Hb as [<- | Hb].
  - exact (Hne eq_refl).
  - assert (Hv : v ∈ flat_map g l)
      by (apply in_flat_map; exists b; split; assumption).
    exact (nodup_app_apart _ _ v Hnd Hva Hv).
  - assert (Hv : v ∈ flat_map g l)
      by (apply in_flat_map; exists a; split; assumption).
    exact (nodup_app_apart _ _ v Hnd Hvb Hv).
  - exact (IH Hrest Ha Hb Hne v Hva Hvb).
Qed.

Lemma flat_map_nodup_one {A : Type} (g : A -> list positive) (l : list A)
    (a : A) :
  NoDup (flat_map g l) -> a ∈ l -> NoDup (g a).
Proof.
  induction l as [ | c l IH]; intros Hnd Ha; [ destruct Ha | ].
  simpl in Hnd. destruct Ha as [<- | Ha].
  - exact (NoDup_app_remove_r _ _ Hnd).
  - exact (IH (NoDup_app_remove_l _ _ Hnd) Ha).
Qed.

Lemma field_variables_nodup : NoDup (field_variables M).
Proof. apply distinct_NoDup. exact M_distinct. Qed.

(* The variables of the fields of one unboxed variable are distinct, and
   apart from those of another. *)
Lemma leaves_nodup (x : variable) (fields : unboxed_fields) :
  fields_of M x = Some fields -> NoDup (leaf_variables fields).
Proof.
  intros Hx. apply fields_of_in in Hx.
  exact (flat_map_nodup_one (fun '(_, fields) => leaf_variables fields) M
           _ field_variables_nodup Hx).
Qed.

Lemma leaves_apart (x x' : variable) (fields fields' : unboxed_fields)
    (v : variable) :
  fields_of M x = Some fields -> fields_of M x' = Some fields' -> x <> x' ->
  v ∈ leaf_variables fields -> v ∈ leaf_variables fields' -> False.
Proof.
  intros Hx Hx' Hne. apply fields_of_in in Hx, Hx'.
  apply (flat_map_nodup_apart (fun '(_, fields) => leaf_variables fields) M
           _ _ field_variables_nodup Hx Hx').
  intros [= Heq _]. exact (Hne Heq).
Qed.

End Fields.

(* ======================================================================== *)
(* Related environments                                                     *)
(* ======================================================================== *)

Section Envs.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Lemma plain_simple_eval (vars vars' : env) (s : simple) :
  env_rel M vars vars' -> plain_simple M s = true ->
  eval_simple vars s = eval_simple vars' s.
Proof.
  intros [Hplain _] Hs. destruct s as [x | sym | c]; simpl; try reflexivity.
  exact (Hplain x Hs).
Qed.

Lemma plain_simples_eval (vars vars' : env) (args : list simple) :
  env_rel M vars vars' -> forallb (plain_simple M) args = true ->
  eval_simples vars args = eval_simples vars' args.
Proof.
  intros Hrel. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
  rewrite (plain_simple_eval vars vars' s Hrel Hs), (IH Hargs).
  reflexivity.
Qed.

Lemma named_plain_eval (statics : static_env) (vars vars' : env) (d : named) :
  env_rel M vars vars' -> named_plain M d = true ->
  eval_named statics vars d = eval_named statics vars' d.
Proof.
  intros Hrel Hd. destruct d as [s | p args | | | | | ]; simpl in Hd |- *;
    try discriminate Hd.
  - exact (plain_simple_eval vars vars' s Hrel Hd).
  - rewrite (plain_simples_eval vars vars' args Hrel Hd). reflexivity.
Qed.

(* With an empty heap, a [Let] of no primitive of the heap runs as
   [eval_named] says: a location among the values of its arguments would
   be one that the heap does not bind. *)
Lemma exec_named_empty_heap (statics : static_env) (rcd : read_record)
    (vars : env) (d : named) (w : value) (hp1 : heap) (rcd1 : read_record) :
  (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
  exec_named statics empty_heap rcd vars d = Some (w, hp1, rcd1) ->
  eval_named statics vars d = Some w ∧ hp1 = empty_heap ∧ rcd1 = rcd.
Proof.
  intros Hnot Hx.
  destruct d as [s | p args | | | | | ]; cbn [exec_named] in Hx.
  2: { destruct (eval_simples vars args) as [vs | ] eqn:Hvs;
         [ | discriminate Hx ].
       cbn in Hx.
       destruct (exec_prim statics empty_heap p vs) as [[v hp'] | ] eqn:He;
         [ | discriminate Hx ].
       cbn in Hx. injection Hx as <- <- <-.
       assert (Hnoloc : ∀ l : location, Value_location l ∉ vs).
       { intros l Hin. unfold exec_prim in He.
         destruct (forallb (argument_bound statics empty_heap) vs) eqn:Hb;
           [ | discriminate He ].
         rewrite forallb_forall in Hb. specialize (Hb _ Hin).
         cbv in Hb. discriminate Hb. }
       assert (Hp : ∀ q : heap_primitive, p <> Heap q)
         by (intros q ->; exact (Hnot q args eq_refl)).
       rewrite (FlambdaHeapProofs.exec_prim_no_location statics empty_heap p vs
         Hp
                 Hnoloc) in He.
       cbn [eval_named]. rewrite Hvs. cbn.
       destruct (eval_prim statics p vs) as [v' | ]; cbn in He;
         [ | discriminate He ].
       injection He as <- <-. auto. }
  all: lazymatch type of Hx with
       | context [eval_named ?st ?vs ?dd] =>
         destruct (eval_named st vs dd) as [v | ] eqn:Hev
       end; cbn in Hx; [ | discriminate Hx ];
       injection Hx as <- <- <-; auto.
Qed.

Lemma rebuild_named_not_heap (sol : solution) (en : rebuild_env)
    (x : variable) (d : named) :
  (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
  ∀ (q : heap_primitive) (args : list simple),
    rebuild_named sol en x d <> Prim (Heap q) args.
Proof.
  intros Hnot q args' H.
  destruct d as [s | p args | | | | | ]; cbn [rebuild_named] in H;
    try discriminate H.
  destruct (FlambdaReaperRebuildProofs.rebuild_named_prim sol en x p (Heap q)
              args args' H) as [Hp _].
  subst p. exact (Hnot q args eq_refl).
Qed.

Lemma named_plain_exec (statics : static_env) (hp : heap) (rcd : read_record)
    (vars vars' : env) (d : named) :
  env_rel M vars vars' -> named_plain M d = true ->
  exec_named statics hp rcd vars d = exec_named statics hp rcd vars' d.
Proof.
  intros Hrel Hd. destruct d as [s | p args | | | | | ]; simpl in Hd |- *;
    try discriminate Hd.
  - rewrite (plain_simple_eval vars vars' s Hrel Hd). reflexivity.
  - rewrite (plain_simples_eval vars vars' args Hrel Hd). reflexivity.
Qed.

Lemma local_bindings_eval (vars vars' : env) (bindings : list static_binding) :
  env_rel M vars vars' -> forallb (local_binding M) bindings = true ->
  ∀ statics : static_env,
    eval_static_bindings vars bindings statics
    = eval_static_bindings vars' bindings statics.
Proof.
  intros Hrel. induction bindings as [ | binding bindings IH]; simpl;
    [ reflexivity | ].
  intros Hb statics. apply andb_true_iff in Hb as [Hb Hbs].
  destruct binding as [id c | id | sym [ | | tag fields | | | ]];
    simpl in Hb; try discriminate Hb.
  destruct (lookup sym (static_values statics)); [ reflexivity | ].
  simpl. rewrite (plain_simples_eval vars vars' fields Hrel Hb).
  destruct (eval_simples vars' fields); simpl; [ apply IH, Hbs | reflexivity ].
Qed.

Lemma env_rel_bind (vars vars' : env) (x : variable) (w : value) :
  env_rel M vars vars' -> plain_variable M x = true ->
  env_rel M ((x, w) :: vars) ((x, w) :: vars').
Proof.
  intros [Hplain Hfields] Hx. split.
  - intros z Hz. rewrite !lookup_cons. destruct (Pos.eqb z x);
      [ reflexivity | exact (Hplain z Hz) ].
  - intros y fields w0 Hy Hlookup p v k Hp.
    rewrite lookup_other in Hlookup
      by (intros ->; exact (plain_not_unboxed M x fields Hx Hy)).
    destruct (Hfields y fields w0 Hy Hlookup p v k Hp) as (w' & Hw' & Hv & Hk).
    exists w'. split; [ exact Hw' | split; [ | exact Hk ] ].
    rewrite lookup_other; [ exact Hv | ]. intros ->.
    apply plain_variable_iff in Hx as [_ Hx].
    pose proof (field_variables_in M y fields x Hy
                  (upath_leaf_in fields p x k Hp)) as Hfv.
    apply among_In in Hfv. rewrite Hfv in Hx. discriminate.
Qed.

Lemma env_rel_empty : env_rel M [] [].
Proof.
  split; [ reflexivity | ]. intros x fields w _ Hlookup.
  discriminate Hlookup.
Qed.

Lemma resolve_rel (K : finite_map (list (option unboxed_fields)))
    (conts conts' : cont_env) (k : continuation) :
  conts_rel M K conts conts' ->
  (lookup k K = None ∧ resolve k conts = Outer k
   ∧ resolve k conts' = Outer k)
  ∨ ∃ sh : list (option unboxed_fields),
      lookup k K = Some sh ∧ cont_rel M sh (resolve k conts) (resolve k conts').
Proof.
  intros Hrel. unfold resolve.
  induction Hrel as [ | k' sh c c' K conts conts' Hc _ IH]; simpl;
    [ left; split; [ reflexivity | split; reflexivity ] | ].
  rewrite ?lookup_cons. destruct (Pos.eqb k k').
  - right. exists sh. split; [ reflexivity | exact Hc ].
  - exact IH.
Qed.

Lemma rec_closures_rel_from
    (KK K S S0 : finite_map (list (option unboxed_fields)))
    (all all' hs hs' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  KK = S ++ K ->
  ub_handlers M KK all all' S -> ub_handlers M KK hs hs' S0 ->
  env_rel M vars vars' -> conts_rel M K conts conts' ->
  conts_rel M (S0 ++ K)
    (map (fun '(k, h) => (k, Closure h vars conts all)) hs ++ conts)
    (map (fun '(k, h) => (k, Closure h vars' conts' all')) hs' ++ conts').
Proof.
  intros HKK Hall Hhs Hvars Hconts.
  induction Hhs as [ | KK0 k sh ps ps' md md' h h' hs hs' S1 Hps Hmd Hmd' Hh
                       _ IH];
    simpl; [ exact Hconts | ].
  constructor; [ | exact (IH HKK Hall) ].
  subst KK0. econstructor; eassumption.
Qed.

Lemma rec_closures_rel (K S : finite_map (list (option unboxed_fields)))
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  ub_handlers M (S ++ K) handlers handlers' S -> env_rel M vars vars' ->
  conts_rel M K conts conts' ->
  conts_rel M (S ++ K) (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hh.
  exact (rec_closures_rel_from (S ++ K) K S S _ _ _ _ vars vars' conts conts'
           eq_refl Hh Hh).
Qed.

End Envs.

(* ======================================================================== *)
(* The shapes of the continuations that are not free                        *)
(* ======================================================================== *)

Scheme ub_mut := Induction for ub Sort Prop
  with ub_handlers_mut := Induction for ub_handlers Sort Prop.

Section Agree.

Variable M : list (node * unboxed_fields).

Definition agree_on (l : list continuation)
    (K K' : finite_map (list (option unboxed_fields))) : Prop :=
  ∀ k : continuation, k ∈ l -> lookup k K' = lookup k K.

Lemma in_without_iff (a : positive) (bound names : list positive) :
  a ∈ without bound names <-> a ∈ names ∧ a ∉ bound.
Proof.
  unfold without. rewrite filter_In, negb_true_iff.
  split; intros [Hin Hbound]; (split; [ exact Hin | ]).
  - intros Ha. apply among_In in Ha. congruence.
  - destruct (among a bound) eqn:Hamong; [ | reflexivity ].
    apply among_In in Hamong. contradiction.
Qed.

Lemma ub_handlers_keys (K : finite_map (list (option unboxed_fields)))
    (hs hs' : list (continuation * continuation_handler))
    (S : finite_map (list (option unboxed_fields))) :
  ub_handlers M K hs hs' S -> map fst S = map fst hs.
Proof.
  induction 1 as [ | K k sh ps ps' md md' h h' hs hs' S _ _ _ _ _ IH];
    simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma lookup_keys {A : Type} (k : positive) (l : list (positive * A)) :
  k ∈ map fst l -> lookup k l <> None.
Proof.
  induction l as [ | [k' a] l IH]; intros Hin; [ destruct Hin | ].
  rewrite lookup_cons. destruct (Pos.eqb k k') eqn:E; [ discriminate | ].
  apply IH. destruct Hin as [Heq | Hin]; [ | exact Hin ].
  simpl in Heq. subst k'. rewrite Pos.eqb_refl in E. discriminate.
Qed.

Lemma agree_app (S K K' : finite_map (list (option unboxed_fields)))
    (bound l : list continuation) :
  map fst S = bound -> agree_on (without bound l) K K' ->
  agree_on l (S ++ K) (S ++ K').
Proof.
  intros HS Hagree k Hk. rewrite !lookup_app.
  destruct (lookup k S) eqn:E; [ reflexivity | ].
  apply Hagree, in_without_iff. split; [ exact Hk | ].
  intros Hin. rewrite <- HS in Hin. exact (lookup_keys k S Hin E).
Qed.

(* The shapes of the continuations not free in a term do not matter. *)
Lemma ub_agree :
  ∀ (K : finite_map (list (option unboxed_fields))) (p q : expr),
    ub M K p q ->
    ∀ K' : finite_map (list (option unboxed_fields)),
      agree_on (free_continuations_of_expr p) K K' -> ub M K' p q.
Proof.
  apply (ub_mut M
           (fun K p q _ =>
              ∀ K', agree_on (free_continuations_of_expr p) K K' ->
                    ub M K' p q)
           (fun K hs hs' S _ =>
              ∀ K',
                agree_on
                  (flat_map (fun '(_, h) => free_continuations_of_handler h)
                     hs)
                  K K' ->
                ub_handlers M K' hs hs' S)).
  - intros K x kind d body body' Hx Hd Hnh _ IH K' Hagree.
    apply ub_let; [ exact Hx | exact Hd | exact Hnh | apply IH, Hagree ].
  - intros K x d fields l body body' Hf Hxf Hl Hoff _ IH K' Hagree.
    apply ub_alloc with (fields := fields); try assumption.
    apply IH, Hagree.
  - intros K x y fields fields_y ls body body' Hf Hxf Hy Hne Hl _ IH K'
      Hagree.
    apply (ub_alias M K' x y fields fields_y ls); try assumption.
    apply IH, Hagree.
  - intros K x kind d f s y fields fields_y nested l body body' Hf Hxf Hr Hs
      Hy Hne Hin Hl _ IH K' Hagree.
    eapply ub_load; try eassumption. apply IH, Hagree.
  - intros K x kind d f s y v fields body body' Hx Hr Hs Hf Hin _ IH K'
      Hagree.
    eapply ub_read; try eassumption. apply IH, Hagree.
  - intros K bindings body body' Hb _ IH K' Hagree.
    apply ub_static; [ exact Hb | ]. apply IH.
    intros k Hk. apply Hagree. simpl. apply in_or_app. right. exact Hk.
  - intros K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' _ IHh _ IHb K'
      Hagree.
    apply (ub_let_cont M K' k sh ps ps' md md' h h' body body'); try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      left. exact Hk.
    + apply IHb. intros k' Hk. rewrite !lookup_cons.
      destruct (Pos.eqb k' k) eqn:E; [ reflexivity | ].
      apply Hagree. simpl. apply in_or_app. right. apply in_without_iff.
      split; [ exact Hk | ]. intros [Heq | []]. subst k'.
      rewrite Pos.eqb_refl in E. discriminate.
  - intros K S hs hs' body body' Hhs IHhs Hst _ IHb K' Hagree.
    pose proof (ub_handlers_keys _ _ _ _ Hhs) as Hkeys.
    apply (ub_let_cont_rec M K' S); [ apply IHhs | exact Hst | apply IHb ];
      apply (agree_app S K K' (map fst hs)); try exact Hkeys;
      intros k Hk; apply Hagree; simpl; apply in_without_iff;
      apply in_without_iff in Hk as [Hk Hnot]; (split; [ | exact Hnot ]);
      apply in_or_app; [ left | right ]; exact Hk.
  - intros K ac ac' (Hk & Htrap & Hargs) K' Hagree.
    apply ub_apply_cont. split; [ exact Hk | split; [ exact Htrap | ] ].
    unfold shapes_at. rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)).
    exact Hargs.
  - intros K scrutinee arms arms' Hs Harms K' Hagree.
    apply ub_switch; [ exact Hs | ]. simpl in Hagree. clear Hs.
    induction Harms as [ | [d ac] [d' ac'] arms arms' [Hd (Hk & Htrap & Hargs)]
                         _ IH];
      constructor.
    + split; [ exact Hd | ]. split; [ exact Hk | split; [ exact Htrap | ] ].
      simpl in Hk, Htrap, Hargs |- *. unfold shapes_at in Hargs |- *.
      rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)). exact Hargs.
    + apply IH. intros k Hin. apply Hagree. simpl. right. apply in_or_app.
      right. exact Hin.
  - intros K K' _. apply ub_invalid.
  - intros K K' _. apply ub_handlers_nil.
  - intros K k sh ps ps' md md' h h' hs hs' S Hps Hmd Hmd' _ IHh _ IHhs K'
      Hagree.
    apply ub_handlers_cons; try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app. left.
      exact Hk.
    + apply IHhs. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      right. exact Hk.
Qed.

End Agree.

(* ======================================================================== *)
(* What the unboxing makes                                                  *)
(* ======================================================================== *)

Definition lets_variable (vks : variable * flambda_kind * simple) : variable :=
  let '(v, _, _) := vks in v.

(* The [Let]s run, each binding its variable to the value of its simple,
   which reads none of the variables bound. *)
Lemma lets_run (l : list (variable * flambda_kind * simple)) :
  ∀ (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rcd : read_record),
    NoDup (map lets_variable l) ->
    (∀ (v : variable) (kind : flambda_kind) (s : simple) (v' : variable),
       (v, kind, s) ∈ l -> v' ∈ map lets_variable l -> s <> Var v') ->
    (∀ (v : variable) (kind : flambda_kind) (s : simple),
       (v, kind, s) ∈ l ->
       ∃ w : value, eval_simple vars s = Some w ∧ kind_of_value w = kind) ->
    ∃ vars1 : env,
      run (length l) (State (lets l body) vars conts statics hp rcd)
      = Step (State body vars1 conts statics hp rcd)
      ∧ (∀ z : variable,
           z ∉ map lets_variable l -> lookup z vars1 = lookup z vars)
      ∧ (∀ (v : variable) (kind : flambda_kind) (s : simple) (w : value),
           (v, kind, s) ∈ l -> eval_simple vars s = Some w ->
           lookup v vars1 = Some w).
Proof.
  induction l as [ | [[v kind] s] l IH]; intros body vars conts statics hp
    rcd Hnd Hread Heval.
  - exists vars. split; [ reflexivity | ]. split; [ reflexivity | ].
    intros v kind s w [].
  - simpl in Hnd. inversion Hnd as [ | v0 l0 Hv Hnd' [Heq0 Heq1]]; subst.
    destruct (IH (Let v kind (Simple s) body) vars conts statics hp rcd
                Hnd')
      as (vars1 & Hrun & Hother & Hbound).
    { intros v1 kind1 s1 v' Hin Hv'. apply (Hread v1 kind1 s1 v');
        [ right; exact Hin | right; exact Hv' ]. }
    { intros v1 kind1 s1 Hin. apply (Heval v1 kind1 s1). right. exact Hin. }
    destruct (Heval v kind s (or_introl eq_refl)) as (w & Hw & Hkind).
    assert (Hs : eval_simple vars1 s = Some w).
    { rewrite <- Hw. destruct s as [z | sym | c]; simpl; try reflexivity.
      apply Hother. intros Hz.
      exact (Hread v kind (Var z) z (or_introl eq_refl) (or_intror Hz)
               eq_refl). }
    exists ((v, w) :: vars1). split; [ | split ].
    + change (lets ((v, kind, s) :: l) body)
        with (lets l (Let v kind (Simple s) body)).
      simpl length.
      rewrite <- Nat.add_1_r, run_add, Hrun, run_succ, step_let_simple.
      rewrite Hs, Hkind, kind_eqb_refl. reflexivity.
    + intros z Hz. rewrite lookup_other by (intros ->; apply Hz; left;
        reflexivity).
      apply Hother. intros Hin. apply Hz. right. exact Hin.
    + intros v' kind' s' w' [Heq | Hin] Hw'.
      * injection Heq as -> -> ->. rewrite Hw in Hw'. injection Hw' as <-.
        apply lookup_head.
      * rewrite lookup_other; [ exact (Hbound v' kind' s' w' Hin Hw') | ].
        intros ->. apply Hv. apply in_map_iff. exists (v, kind', s').
        split; [ reflexivity | exact Hin ].
Qed.

Lemma is_number_of_kind_kind (bn : boxable_number) (c : const) :
  is_number_of_kind bn c = true -> kind_of_const c = unboxed_kind bn.
Proof.
  destruct bn, c; simpl; intros H; try discriminate H; reflexivity.
Qed.

(* The variable of a field of an allocation that is made is bound to the
   field. *)
Lemma alloc_field_eval (statics : static_env) (vars : env) (d : named)
    (w : value) (f : field) (kind : flambda_kind) (s : simple) :
  eval_named statics vars d = Some w -> alloc_field d f kind s ->
  ∃ w' : value,
    eval_simple vars s = Some w' ∧ field_value f w = Some w'
    ∧ kind_of_value w' = kind.
Proof.
  intros Hd Hf. destruct d as [ | p args | | | | | ]; try contradiction.
  destruct p as [p | p | p | p | p]; try contradiction.
  - destruct p; try contradiction.
    destruct args as [ | c [ | ]]; try contradiction.
    destruct Hf as (-> & -> & ->). simpl in Hd.
    destruct (eval_simple vars c) as [v | ] eqn:Hc; [ | discriminate Hd ].
    simpl in Hd. destruct v as [c0 | | | | | | | | ]; try discriminate Hd.
    rewrite eval_prim_unary in Hd. cbn [eval_unary_primitive] in Hd.
    destruct (is_number_of_kind _ c0) eqn:Hk; [ | discriminate Hd ].
    injection Hd as <-. exists (Value_const c0).
    split; [ reflexivity | split; [ reflexivity | ] ].
    exact (is_number_of_kind_kind _ c0 Hk).
  - destruct p as [[tag] | ]; try contradiction.
    destruct Hf as (i & -> & -> & Hnth). simpl in Hd.
    destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate Hd ].
    simpl in Hd. rewrite eval_prim_variadic in Hd.
    cbn [eval_variadic_primitive] in Hd.
    destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs
              && forallb (bound_if_symbol statics) vs) eqn:Hok;
      [ | discriminate Hd ].
    injection Hd as <-. apply andb_true_iff in Hok as [Hkinds _].
    destruct (nth_error vs i) as [w' | ] eqn:Hw'.
    + exists w'. split; [ | split; [ exact Hw' | ] ].
      * destruct (FlambdaReaperFlowProofs.eval_simples_nth vars args vs i w'
                    Hargs Hw')
          as (s' & Hs' & Heval). rewrite Hnth in Hs'. injection Hs' as <-.
        exact Heval.
      * rewrite forallb_forall in Hkinds.
        apply kind_eqb_eq, Hkinds. exact (nth_error_In _ _ Hw').
    + exfalso. apply nth_error_None in Hw'.
      apply FlambdaReaperFlowProofs.eval_simples_length in Hargs.
      assert (Hlt : i < length args)
        by (apply nth_error_Some; rewrite Hnth; discriminate).
      lia.
Qed.

(* A read of a field: of the field of the value read, which is not a
   constant. *)
Lemma read_eval (statics : static_env) (vars : env) (d : named) (f : field)
    (s : simple) (r : value) :
  read_of d = Some (f, s) -> eval_named statics vars d = Some r ->
  ∃ w0 : value,
    eval_simple vars s = Some w0 ∧ (∀ c : const, w0 <> Value_const c)
    ∧ ∀ w' : value, field_value f w0 = Some w' -> r = w'.
Proof.
  intros Hread Hd.
  destruct d as [ | p args | | | | | ]; try discriminate Hread.
  destruct p as [p | p | p | p | p]; try discriminate Hread.
  destruct args as [ | a [ | ]]; destruct p; try discriminate Hread;
    simpl in Hread; injection Hread as <- <-; simpl in Hd;
    destruct (eval_simple vars a) as [w0 | ]; try discriminate Hd;
    simpl in Hd; rewrite eval_prim_unary in Hd;
    cbn [eval_unary_primitive] in Hd; exists w0; (split; [ reflexivity | ]).
  - (* Block_load *)
    destruct kind as [tag size field_kind]. unfold eval_block_load in Hd.
    destruct (block_of statics w0) as [[tag0 fields] | ] eqn:Hblock;
      [ | discriminate Hd ].
    simpl in Hd. split.
    + intros c ->. discriminate Hblock.
    + intros w' Hw'.
      destruct w0 as [ | | | | tagw fieldsw | | | | ]; try discriminate Hw'.
      simpl in Hw'. simpl in Hblock. injection Hblock as Htag Hfields.
      subst tagw fieldsw.
      destruct (known_is tag tag0
                && known_is size (Z.of_nat (length fields))
                && (0 <=? field)%Z); [ | discriminate Hd ].
      rewrite Hw' in Hd.
      destruct (match field_kind with
                | Any_value => kind_eqb (kind_of_value w') Value
                | Immediate => is_tagged_immediate w'
                end); [ | discriminate Hd ].
      injection Hd as <-. reflexivity.
  - (* Unbox_number *)
    destruct (boxed_number_of statics w0) as [c | ] eqn:Hbox;
      [ | discriminate Hd ].
    simpl in Hd. split.
    + intros c' ->. discriminate Hbox.
    + intros w' Hw'.
      destruct w0 as [ | | | | | | c0 | | ]; try discriminate Hw'.
      simpl in Hw'. injection Hw' as <-. simpl in Hbox.
      unfold boxable_contents in Hbox.
      destruct (boxable_number_of_const c0); [ | discriminate Hbox ].
      injection Hbox as <-.
      destruct (is_number_of_kind _ c0); [ | discriminate Hd ].
      injection Hd as <-. reflexivity.
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Section Simulation.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Lemma forall2_in_both {A B : Type} (P : A -> B -> Prop) (l : list A)
    (l' : list B) (a : A) :
  Forall2 P l l' -> a ∈ l -> ∃ b : B, b ∈ l' ∧ P a b.
Proof.
  induction 1 as [ | a0 b0 l l' H _ IH]; intros Hin; [ destruct Hin | ].
  destruct Hin as [<- | Hin]; [ exists b0; split; [ left | ]; auto | ].
  destruct (IH Hin) as (b & Hb & Hab). exists b. split; [ right | ]; auto.
Qed.

Lemma forall2_concat_in {A B : Type} (P : A -> list B -> Prop) (l : list A)
    (ls : list (list B)) (y : B) :
  Forall2 P l ls -> y ∈ concat ls ->
  ∃ (a : A) (l0 : list B), a ∈ l ∧ P a l0 ∧ y ∈ l0.
Proof.
  induction 1 as [ | a0 l0 l ls H _ IH]; intros Hin; [ destruct Hin | ].
  cbn [concat] in Hin. apply in_app_or in Hin as [Hin | Hin].
  - exists a0, l0. split; [ left; reflexivity | split; assumption ].
  - destruct (IH Hin) as (a & l1 & Ha & Hal & Hy).
    exists a, l1. split; [ right; exact Ha | split; assumption ].
Qed.

Lemma in_pairs_lets (pairs : list (variable * variable * flambda_kind))
    (v : variable) (kind : flambda_kind) (s : simple) :
  (v, kind, s) ∈ pairs_lets pairs ->
  ∃ v' : variable, s = Var v' ∧ (v, v', kind) ∈ pairs.
Proof.
  unfold pairs_lets. intros Hin. apply in_map_iff in Hin
    as ([[x x'] k] & Heq & Hin). injection Heq as -> -> <-.
  exists x'. split; [ reflexivity | exact Hin ].
Qed.

Lemma pairs_in_lets (pairs : list (variable * variable * flambda_kind))
    (v v' : variable) (kind : flambda_kind) :
  (v, v', kind) ∈ pairs -> (v, kind, Var v') ∈ pairs_lets pairs.
Proof.
  intros Hin. unfold pairs_lets.
  apply (in_map (fun '(x, x', kind) => (x, kind, Var x')) _ _ Hin).
Qed.

Lemma pairs_lets_variables (pairs : list (variable * variable * flambda_kind))
    : map lets_variable (pairs_lets pairs) = map (fun '(x, _, _) => x) pairs.
Proof.
  unfold pairs_lets. rewrite map_map. apply map_ext.
  intros [[x x'] k]. reflexivity.
Qed.

(* The fields of a value that a variable has, read through the pairs of
   another variable's fields alike. *)
Lemma pairs_hold (vars' : env) (fields fields_y : unboxed_fields)
    (w : value) (pairs : list (variable * variable * flambda_kind)) :
  unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
  = Some pairs ->
  fields_hold vars' fields_y w ->
  map (fun '(x, _, _) => x) pairs = leaf_variables fields
  ∧ (∀ (v v' : variable) (k : flambda_kind),
       (v, v', k) ∈ pairs ->
       v' ∈ leaf_variables fields_y
       ∧ ∃ w' : value, lookup v' vars' = Some w' ∧ kind_of_value w' = k)
  ∧ ∀ (p : list field) (v : variable) (k : flambda_kind),
      upath Value (Unboxed fields) p v k ->
      ∃ (v' : variable) (w' : value),
        (v, v', k) ∈ pairs ∧ lookup v' vars' = Some w'
        ∧ path_value p w = Some w'.
Proof.
  intros Hp Hhold.
  destruct (pairs_spec (Unboxed fields) Value (Unboxed fields_y) pairs Hp)
    as (Ha & Hb & Hc & _).
  split; [ exact Hc | split ].
  - intros v v' k Hin. destruct (Ha v v' k Hin) as (p & _ & Hp').
    split; [ exact (upath_leaf_in fields_y p v' k Hp') | ].
    destruct (Hhold p v' k Hp') as (w' & _ & Hl & Hk).
    exists w'. split; assumption.
  - intros p v k Hpath. destruct (Hb p v k Hpath) as (v' & Hin & Hp').
    destruct (Hhold p v' k Hp') as (w' & Hpw & Hl & _).
    exists v', w'. split; [ exact Hin | split; assumption ].
Qed.

(* A field unboxed in turn holds what is at its paths. *)
Lemma fields_hold_nested (vars' : env) (fields_y nested : unboxed_fields)
    (f : field) (w0 r : value) :
  fields_hold vars' fields_y w0 -> (f, Unboxed nested) ∈ fields_y ->
  (∀ w' : value, field_value f w0 = Some w' -> r = w') ->
  fields_hold vars' nested r.
Proof.
  intros Hhold Hin Hfield p v k Hpath.
  assert (Hp : upath Value (Unboxed fields_y) (f :: p) v k).
  { econstructor; [ exact Hin | ].
    exact (upath_unboxed_kind Value (field_kind f) nested p v k Hpath). }
  destruct (Hhold _ v k Hp) as (w' & Hpw & Hl & Hk).
  cbn [path_value] in Hpw.
  destruct (field_value f w0) as [w1 | ] eqn:Hw1; [ | discriminate Hpw ].
  rewrite <- (Hfield w1 eq_refl) in Hpw.
  exists w'. split; [ exact Hpw | split; assumption ].
Qed.

(* An unboxed variable bound, the variables of its fields bound to the
   values at their paths in its value. *)
Lemma env_rel_bind_fields (vars vars' vars1 : env) (x : variable)
    (fields : unboxed_fields) (w : value) :
  env_rel M vars vars' -> fields_of M x = Some fields ->
  (∀ z : variable,
     z ∉ leaf_variables fields -> lookup z vars1 = lookup z vars') ->
  fields_hold vars1 fields w ->
  env_rel M ((x, w) :: vars) vars1.
Proof.
  intros [Hplain Hfs] Hx Hother Hhold. split.
  - intros z Hz. rewrite lookup_other
      by (intros ->; exact (plain_not_unboxed M x fields Hz Hx)).
    rewrite (Hplain z Hz), Hother; [ reflexivity | ].
    intros Hin. apply plain_variable_iff in Hz as [_ Hz].
    pose proof (field_variables_in M x fields z Hx Hin) as Hfv.
    apply among_In in Hfv. rewrite Hfv in Hz. discriminate.
  - intros y fy w0 Hy Hlookup p v k Hp.
    destruct (Pos.eq_dec y x) as [-> | Hne].
    + rewrite lookup_head in Hlookup. injection Hlookup as <-.
      rewrite Hx in Hy. injection Hy as <-. exact (Hhold p v k Hp).
    + rewrite lookup_other in Hlookup by exact Hne.
      destruct (Hfs y fy w0 Hy Hlookup p v k Hp) as (w' & Hpw & Hl & Hk).
      exists w'. split; [ exact Hpw | split; [ | exact Hk ] ].
      rewrite Hother; [ exact Hl | ]. intros Hvx.
      exact (leaves_apart M M_distinct y x fy fields v Hy Hx Hne
               (upath_leaf_in fy p v k Hp) Hvx).
Qed.

(* The [Let]s of the variables of the fields of an unboxed variable run,
   and bind them as the relation asks. *)
Lemma lets_case (vars vars' : env) (x : variable) (fields : unboxed_fields)
    (w : value) (l : list (variable * flambda_kind * simple)) (body : expr)
    (conts : cont_env) (statics : static_env) (hp : heap) (rcd : read_record) :
  env_rel M vars vars' -> fields_of M x = Some fields ->
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
    run (length l) (State (lets l body) vars' conts statics hp rcd)
    = Step (State body vars1 conts statics hp rcd)
    ∧ env_rel M ((x, w) :: vars) vars1.
Proof.
  intros Hvars Hx Hvl Hread Heval Hpaths.
  destruct (lets_run l body vars' conts statics hp rcd)
    as (vars1 & Hrun & Hother & Hbound).
  - rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hx).
  - intros v kind s v' Hin Hv'. rewrite Hvl in Hv'.
    exact (Hread v kind s v' Hin Hv').
  - exact Heval.
  - exists vars1. split; [ exact Hrun | ].
    apply (env_rel_bind_fields vars vars' vars1 x fields w Hvars Hx).
    + intros z Hz. apply Hother. rewrite Hvl. exact Hz.
    + intros p v k Hp. destruct (Hpaths p v k Hp) as (s & w' & Hin & Hs & Hpw).
      destruct (Heval v k s Hin) as (w'' & Hs' & Hk).
      rewrite Hs in Hs'. injection Hs' as <-.
      exists w'. split; [ exact Hpw | ].
      split; [ exact (Hbound v k s w' Hin Hs) | exact Hk ].
Qed.

Lemma run_answer (R : state -> state -> Prop) (s s' t t' : state) (n : nat) :
  run n t = Step t' -> R s' t' ->
  (n = 0 -> measure s' < measure s) ->
  step s' = Undefined
  ∨ (∃ t'' : state, step t = Step t'' ∧ R s' t'')
  ∨ (measure s' < measure s ∧ R s' t)
  ∨ (∃ (t'' : state) (m : nat), run (S (S m)) t = Step t'' ∧ R s' t'').
Proof.
  intros Hrun Hrel Hzero. destruct n as [ | [ | m]].
  - simpl in Hrun. injection Hrun as <-. right. right. left.
    split; [ apply Hzero; reflexivity | exact Hrel ].
  - right. left. exists t'. split; [ | exact Hrel ].
    rewrite run_succ in Hrun.
    destruct (step t) as [t0 | | ]; simpl in Hrun; congruence.
  - right. right. right. exists t', m. split; assumption.
Qed.

Lemma alloc_field_kind (d : named) (f : field) (kind : flambda_kind)
    (s : simple) :
  alloc_field d f kind s -> kind = field_kind f.
Proof.
  intros Hf. destruct d as [ | p args | | | | | ]; try contradiction.
  destruct p as [p | p | p | p | p]; try contradiction.
  - destruct p; try contradiction.
    destruct args as [ | c [ | ]]; try contradiction.
    destruct Hf as (-> & -> & _). reflexivity.
  - destruct p as [[tag] | ]; try contradiction.
    destruct Hf as (i & -> & -> & _). reflexivity.
Qed.

Lemma params_rel_nil_inv (ps ps' : list (variable * flambda_kind)) :
  params_rel M [] ps ps' -> ps = [] ∧ ps' = [].
Proof. intros H. inversion H. split; reflexivity. Qed.

Lemma params_rel_cons_inv (o : option unboxed_fields)
    (sh : list (option unboxed_fields))
    (ps ps' : list (variable * flambda_kind)) :
  params_rel M (o :: sh) ps ps' ->
  ∃ (p : variable) (kind : flambda_kind) ps1, ps = (p, kind) :: ps1.
Proof. intros H. inversion H; subst; eauto. Qed.

Lemma params_rel_none_inv (sh : list (option unboxed_fields))
    (ps ps' : list (variable * flambda_kind)) :
  params_rel M (None :: sh) ps ps' ->
  ∃ (p : variable) (kind : flambda_kind) ps1 ps1',
    ps = (p, kind) :: ps1 ∧ ps' = (p, kind) :: ps1'
    ∧ plain_variable M p = true ∧ params_rel M sh ps1 ps1'.
Proof.
  intros H. inversion H; subst.
  eexists _, _, _, _. repeat split; eassumption.
Qed.

Lemma params_rel_some_inv (sh : list (option unboxed_fields))
    (fields : unboxed_fields) (ps ps' : list (variable * flambda_kind)) :
  params_rel M (Some fields :: sh) ps ps' ->
  ∃ (p : variable) ps1 ps1',
    ps = (p, Value) :: ps1 ∧ ps' = fields_leaves fields ++ ps1'
    ∧ fields_of M p = Some fields ∧ params_rel M sh ps1 ps1'.
Proof.
  intros H. inversion H; subst.
  eexists _, _, _. repeat split; eassumption.
Qed.

Lemma args_rel_nil_inv (args args' : list simple) :
  args_rel M [] args args' ->
  args' = args ∧ forallb (plain_simple M) args = true.
Proof. intros H. inversion H; subst. split; [ reflexivity | assumption ]. Qed.

Lemma bind_params_app (l1 l2 : list variable) (vs1 vs2 : list value)
    (vars vars1 : env) :
  bind_params l1 vs1 vars = Some vars1 ->
  bind_params (l1 ++ l2) (vs1 ++ vs2) vars = bind_params l2 vs2 vars1.
Proof.
  revert vs1 vars. induction l1 as [ | x l1 IH]; intros [ | v vs1] vars Hb;
    simpl in Hb |- *; try discriminate Hb.
  - injection Hb as <-. reflexivity.
  - exact (IH vs1 _ Hb).
Qed.

Lemma in_concat_of {A : Type} (ls : list (list A)) (l : list A) (a : A) :
  l ∈ ls -> a ∈ l -> a ∈ concat ls.
Proof.
  induction ls as [ | l0 ls IH]; intros Hl Ha; [ destruct Hl | ].
  cbn [concat]. apply in_or_app. destruct Hl as [-> | Hl];
    [ left; exact Ha | right; exact (IH Hl Ha) ].
Qed.

Lemma alloc_entries_variables (x : variable) (d : named)
    (fields : unboxed_fields)
    (ls : list (list (variable * flambda_kind * simple))) :
  Forall2 (alloc_entry M x d) fields ls ->
  map lets_variable (concat ls) = leaf_variables fields.
Proof.
  induction 1 as [ | [f u] l fields ls Hfu _ IH]; [ reflexivity | ].
  cbn [concat]. rewrite map_app, IH. unfold leaf_variables.
  rewrite leaves_cons, map_app. f_equal.
  destruct u as [v | nested]; cbn [alloc_entry] in Hfu.
  - destruct Hfu as (kind & s & -> & _). reflexivity.
  - destruct Hfu as (b & fb & pairs & _ & _ & _ & Hpairs & ->).
    rewrite pairs_lets_variables.
    destruct (pairs_spec _ _ _ _ Hpairs) as (_ & _ & Hc & _).
    rewrite Hc. reflexivity.
Qed.

(* The variables of the fields of an argument, in the second run. *)
Lemma pairs_values (vars' : env)
    (pairs : list (variable * variable * flambda_kind)) :
  (∀ (v v' : variable) (k : flambda_kind),
     (v, v', k) ∈ pairs ->
     ∃ w' : value, lookup v' vars' = Some w' ∧ kind_of_value w' = k) ->
  ∃ ws : list value,
    eval_simples vars' (map (fun '(_, x', _) => Var x') pairs) = Some ws
    ∧ map kind_of_value ws = map (fun '(_, _, k) => k) pairs
    ∧ Forall2
        (fun (vvk : variable * variable * flambda_kind) (w : value) =>
           let '(_, x', _) := vvk in lookup x' vars' = Some w)
        pairs ws.
Proof.
  induction pairs as [ | [[v v'] k] pairs IH]; intros Hp.
  - exists []. split; [ reflexivity | split; [ reflexivity | constructor ] ].
  - destruct (Hp v v' k (or_introl eq_refl)) as (w' & Hl & Hk).
    destruct (IH (fun a b c Hin => Hp a b c (or_intror Hin)))
      as (ws & Hws & Hks & Hall).
    exists (w' :: ws). split; [ | split ].
    + cbn [map eval_simples eval_simple]. rewrite Hl, Hws. reflexivity.
    + simpl. rewrite Hk, Hks. reflexivity.
    + constructor; assumption.
Qed.

(* The variables of the fields of a parameter bound to those values. *)
Lemma bind_pairs (vars' : env)
    (pairs : list (variable * variable * flambda_kind)) (ws : list value) :
  Forall2
    (fun (vvk : variable * variable * flambda_kind) (w : value) =>
       let '(_, x', _) := vvk in lookup x' vars' = Some w)
    pairs ws ->
  (∀ vars : env,
     ∃ vars1 : env,
       bind_params (map (fun '(x, _, _) => x) pairs) ws vars = Some vars1)
  ∧ ∀ vars vars1 : env,
      NoDup (map (fun '(x, _, _) => x) pairs) ->
      bind_params (map (fun '(x, _, _) => x) pairs) ws vars = Some vars1 ->
      (∀ z : variable,
         z ∉ map (fun '(x, _, _) => x) pairs ->
         lookup z vars1 = lookup z vars)
      ∧ ∀ (v v' : variable) (k : flambda_kind) (w : value),
          (v, v', k) ∈ pairs -> lookup v' vars' = Some w ->
          lookup v vars1 = Some w.
Proof.
  induction 1 as [ | [[v v'] k] w0 pairs ws Hw0 _ [IHb IH]].
  - split; [ intros vars; exists vars; reflexivity | ].
    intros vars vars1 _ Hb. simpl in Hb. injection Hb as <-.
    split; [ reflexivity | ]. intros v v' k w [].
  - split.
    + intros vars. exact (IHb ((v, w0) :: vars)).
    + intros vars vars1 Hnd Hb. cbn [map] in Hnd, Hb. simpl in Hb.
      inversion Hnd as [ | v0 l0 Hv Hnd' [Heq0 Heq1]]; subst.
      destruct (IH _ _ Hnd' Hb) as [Hother Hbound]. split.
      * intros z Hz. rewrite Hother
          by (intros Hin; apply Hz; right; exact Hin).
        rewrite lookup_other; [ reflexivity | ].
        intros ->. apply Hz. left. reflexivity.
      * intros x x' k' w [Heq | Hin] Hl.
        -- injection Heq as -> -> ->. rewrite Hl in Hw0. injection Hw0 as ->.
           rewrite (Hother x Hv). apply lookup_head.
        -- exact (Hbound x x' k' w Hin Hl).
Qed.

(* The arguments related, given to parameters related, are bound to
   values related. *)
Lemma bind_args_rel (sh : list (option unboxed_fields))
    (args args' : list simple) :
  args_rel M sh args args' ->
  ∀ (ps ps' : list (variable * flambda_kind)) (vars vars' cv cv' hv : env)
    (vs : list value),
    params_rel M sh ps ps' -> env_rel M vars vars' -> env_rel M cv cv' ->
    eval_simples vars args = Some vs ->
    map kind_of_value vs = map snd ps ->
    bind_params (map fst ps) vs cv = Some hv ->
    ∃ (vs' : list value) (hv' : env),
      eval_simples vars' args' = Some vs'
      ∧ map kind_of_value vs' = map snd ps'
      ∧ bind_params (map fst ps') vs' cv' = Some hv' ∧ env_rel M hv hv'.
Proof.
  induction 1 as [args Hargs | o sh | sh a args args' Ha _ IH
                 | sh fields fields_y y pairs args args' Hy Hpairs _ IH];
    intros ps ps' vars vars' cv cv' hv vs Hps Hvars Hcv Heval Hkinds Hbind.
  - destruct (params_rel_nil_inv ps ps' Hps) as [-> ->].
    simpl in Hbind. destruct vs as [ | w vs]; [ | discriminate Hbind ].
    injection Hbind as <-.
    destruct args as [ | s args];
      [ | simpl in Heval; destruct (eval_simple vars s);
          [ | discriminate Heval ];
          destruct (eval_simples vars args); discriminate Heval ].
    exists [], cv'. split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ reflexivity | exact Hcv ].
  - simpl in Heval. injection Heval as <-.
    destruct (params_rel_cons_inv o sh ps ps' Hps) as (p & kind & ps1 & ->).
    discriminate Hbind.
  - destruct (params_rel_none_inv sh ps ps' Hps)
      as (p & kind & ps1 & ps1' & -> & -> & Hp & Hps1).
    simpl in Heval.
    destruct (eval_simple vars a) as [w | ] eqn:Hwa; [ | discriminate Heval ].
    destruct (eval_simples vars args) as [vs1 | ] eqn:Hargs;
      [ | discriminate Heval ].
    injection Heval as <-. simpl in Hkinds. injection Hkinds as Hk Hkinds.
    simpl in Hbind.
    destruct (IH ps1 ps1' vars vars' ((p, w) :: cv) ((p, w) :: cv') hv vs1
                Hps1 Hvars (env_rel_bind M cv cv' p w Hcv Hp) Hargs Hkinds
                Hbind)
      as (vs' & hv' & Hev' & Hk' & Hb' & Hrel).
    exists (w :: vs'), hv'. split.
    + simpl. rewrite <- (plain_simple_eval M vars vars' a Hvars Ha), Hwa, Hev'.
      reflexivity.
    + split; [ simpl; rewrite Hk, Hk'; reflexivity | ].
      split; [ exact Hb' | exact Hrel ].
  - destruct (params_rel_some_inv sh fields ps ps' Hps)
      as (p & ps1 & ps1' & -> & -> & Hp & Hps1).
    simpl in Heval.
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | discriminate Heval ].
    destruct (eval_simples vars args) as [vs1 | ] eqn:Hargs;
      [ | discriminate Heval ].
    injection Heval as <-. simpl in Hkinds. injection Hkinds as _ Hkinds.
    simpl in Hbind.
    destruct (pairs_hold vars' fields fields_y w pairs Hpairs
                (proj2 Hvars y fields_y w Hy Hw))
      as (Hfirsts & Hseconds & Hpaths).
    destruct (pairs_values vars' pairs
                (fun v v' k Hin => proj2 (Hseconds v v' k Hin)))
      as (ws & Hws & Hwk & Hall).
    destruct (bind_pairs vars' pairs ws Hall) as [Hwb Hbound].
    destruct (Hwb cv') as (cv1 & Hcv1).
    assert (Hnd : NoDup (map (fun '(x, _, _) => x) pairs))
      by (rewrite Hfirsts; exact (leaves_nodup M M_distinct p fields Hp)).
    destruct (Hbound cv' cv1 Hnd Hcv1) as [Hother Hbv].
    assert (Hrel1 : env_rel M ((p, w) :: cv) cv1).
    { apply (env_rel_bind_fields cv cv' cv1 p fields w Hcv Hp).
      - intros z Hz. apply Hother. rewrite Hfirsts. exact Hz.
      - intros path v k Hpath.
        destruct (Hpaths path v k Hpath) as (v' & w' & Hin & Hl & Hpw).
        destruct (Hseconds v v' k Hin) as [_ (w'' & Hl' & Hk)].
        rewrite Hl in Hl'. injection Hl' as <-.
        exists w'. split; [ exact Hpw | ].
        split; [ exact (Hbv v v' k w' Hin Hl) | exact Hk ]. }
    destruct (IH ps1 ps1' vars vars' ((p, w) :: cv) cv1 hv vs1 Hps1 Hvars
                Hrel1 Hargs Hkinds Hbind)
      as (vs' & hv' & Hev' & Hk' & Hb' & Hrel).
    destruct (pairs_spec (Unboxed fields) Value (Unboxed fields_y) pairs
                Hpairs) as (_ & _ & _ & Hthirds).
    exists (ws ++ vs'), hv'. split.
    + rewrite FlambdaSemanticsProofs.eval_simples_app, Hws, Hev'.
      reflexivity.
    + split; [ rewrite !map_app, Hwk, Hk', Hthirds; reflexivity | ].
      split; [ | exact Hrel ].
      rewrite map_app. change (map fst (fields_leaves fields))
        with (leaf_variables fields).
      rewrite <- Hfirsts, (bind_params_app _ _ _ _ _ _ Hcv1). exact Hb'.
Qed.

Lemma find_arm_rel (K : finite_map (list (option unboxed_fields))) (i : Z)
    (arms arms' : list (Z * apply_cont)) :
  Forall2
    (fun (arm arm' : Z * apply_cont) =>
       fst arm' = fst arm ∧ ac_rel M K (snd arm) (snd arm'))
    arms arms' ->
  ∀ ac : apply_cont,
    find_arm i arms = Some ac ->
    ∃ ac' : apply_cont, find_arm i arms' = Some ac' ∧ ac_rel M K ac ac'.
Proof.
  induction 1 as [ | [d ac0] [d' ac0'] arms arms' [Hd Hac] _ IH];
    intros ac Hfind; [ discriminate Hfind | ].
  simpl in Hd, Hac, Hfind |- *. subst d'. destruct (Z.eqb i d).
  - injection Hfind as <-. exists ac0'. split; [ reflexivity | exact Hac ].
  - exact (IH ac Hfind).
Qed.

Lemma read_not_heap (d : named) (f : field) (s : simple) :
  read_of d = Some (f, s) ->
  ∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args.
Proof. intros H q args ->. discriminate H. Qed.

Lemma related_answered (s t : state) :
  related M s t -> answered (related M) eq measure s t.
Proof.
  destruct s as [p vars conts statics hp rcd],
    t as [q vars' conts' statics' hp' rcd'].
  intros (K & Hub & Hvars & Hconts & Hstatics & Hhp & Hrcd & Hempty).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst statics' hp' rcd' hp. unfold answered. revert Hconts.
  destruct Hub as [K x kind d body body' Hx Hd Hnh Hbody
    | K x d fields ls body body' Hfields Hxf Hl Hoff Hbody
    | K x y fields fields_y pairs body body' Hfields Hxf Hy Hne Hl Hbody
    | K x kind d f s y fields fields_y nested pairs body body' Hfields Hxf Hread
        Hs Hy Hne Hin Hl Hbody
    | K x kind d f s y v fields body body' Hx Hread Hs Hfields Hin Hbody
    | K bindings body body' Hbindings Hbody
    | K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' Hh Hbody
    | K S handlers handlers' body body' Hhandlers Hstat Hbody
    | K ac ac' Hac
    | K scrutinee arms arms' Hscrutinee Harms
    | K ]; intros Hconts.
  - (* A [Let] of both. *)
    rewrite step_let_exec,
      (named_plain_exec M statics empty_heap rcd vars vars' d Hvars Hd).
    destruct (exec_named statics empty_heap rcd vars' d) as [[[w hp1] rcd1] | ]
      eqn:Hw; [ | exact I ].
    destruct (exec_named_empty_heap statics rcd vars' d w hp1 rcd1 Hnh Hw)
      as (_ & -> & ->).
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hk; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_exec, Hw, Hk; reflexivity | ].
    exists K. split; [ exact Hbody | ].
    split; [ exact (env_rel_bind M _ _ x w Hvars Hx) | ].
    split; [ exact Hconts | repeat split ].
  - (* The allocation of an unboxed variable. *)
    rewrite step_let_exec.
    destruct (exec_named statics empty_heap rcd vars d) as [[[w hp1] rcd1] | ]
      eqn:Hx; [ | exact I ].
    destruct (exec_named_empty_heap statics rcd vars d w hp1 rcd1 Hoff Hx)
      as (Hw & -> & ->).
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (lets_case vars vars' x fields w (concat ls) body' conts' statics
                empty_heap rcd Hvars Hfields)
      as (vars1 & Hrun & Hrel).
    + exact (alloc_entries_variables x d fields ls Hl).
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
        destruct (alloc_field_eval statics vars d w f kind0 s0 Hw Hf)
          as (w' & Hs & _ & Hkw).
        exists w'. split; [ | exact Hkw ].
        rewrite <- (plain_simple_eval M vars vars' s0 Hvars Hplain). exact Hs.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval statics vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & _ & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 Hvars b fb wb Hb Hs))
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
        destruct (alloc_field_eval statics vars d w f (field_kind f) s0 Hw Hf)
          as (w' & Hs & Hfw & _).
        exists s0, w'. split; [ exact (in_concat_of _ _ _ Hl0 (or_introl
                                                       eq_refl)) | ].
        split; [ rewrite <- (plain_simple_eval M vars vars' s0 Hvars Hplain);
                 exact Hs | ].
        cbn [path_value]. rewrite Hfw. reflexivity.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval statics vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & Hfw & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 Hvars b fb wb Hb Hs))
          as (_ & _ & Hpaths).
        destruct (Hpaths p0 v k (upath_unboxed_kind _ Value _ _ _ _ Hp0))
          as (v' & w' & Hin' & Hl' & Hpw).
        exists (Var v'), w'.
        split; [ exact (in_concat_of _ _ _ Hl0
                          (pairs_in_lets pairs v v' k Hin')) | ].
        split; [ exact Hl' | ]. cbn [path_value]. rewrite Hfw. exact Hpw.
    + apply (run_answer (related M) _ _ _ _ (length (concat ls)) Hrun).
      * exists K. unfold related.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | repeat split ].
      * intros _. unfold measure. simpl. lia.
  - (* An alias of an unboxed variable. *)
    rewrite step_let_simple. cbn [eval_simple].
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (pairs_hold vars' fields fields_y w pairs Hl
                (proj2 Hvars y fields_y w Hy Hw))
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case vars vars' x fields w (pairs_lets pairs) body' conts'
                statics empty_heap rcd Hvars Hfields)
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
    + apply (run_answer (related M) _ _ _ _ (length (pairs_lets pairs)) Hrun).
      * exists K. unfold related.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | repeat split ].
      * intros _. unfold measure. simpl. lia.
  - (* The load of a field unboxed in turn. *)
    rewrite step_let_exec.
    destruct (exec_named statics empty_heap rcd vars d) as [[[r hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap statics rcd vars d r hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r) Value) eqn:Hk; [ | exact I ].
    destruct (read_eval statics vars d f s r Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    pose proof (fields_hold_nested vars' fields_y nested f w0 r
                  (proj2 Hvars y fields_y w0 Hy Hw0) Hin Hfield) as Hhold.
    destruct (pairs_hold vars' fields nested r pairs Hl Hhold)
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case vars vars' x fields r (pairs_lets pairs) body' conts'
                statics empty_heap rcd Hvars Hfields)
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
    + apply (run_answer (related M) _ _ _ _ (length (pairs_lets pairs)) Hrun).
      * exists K. unfold related.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | repeat split ].
      * intros _. unfold measure. simpl. lia.
  - (* A read of a field. *)
    rewrite step_let_exec.
    destruct (exec_named statics empty_heap rcd vars d) as [[[r hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap statics rcd vars d r hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r) kind) eqn:Hk; [ | exact I ].
    destruct (read_eval statics vars d f s r Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    assert (Hp : upath Value (Unboxed fields) [f] v (field_kind f))
      by (econstructor; [ exact Hin | constructor ]).
    destruct (proj2 Hvars y fields w0 Hfields Hw0 [f] v _ Hp)
      as (w' & Hpw & Hv & _).
    cbn [path_value] in Hpw.
    destruct (field_value f w0) as [w1 | ] eqn:Hfw; [ | discriminate Hpw ].
    injection Hpw as <-.
    rewrite <- (Hfield w1 eq_refl) in Hv.
    right. left. eexists. split.
    + rewrite step_let_simple. cbn [eval_simple]. rewrite Hv, Hk.
      reflexivity.
    + exists K. split; [ exact Hbody | ].
      split; [ exact (env_rel_bind M _ _ x r Hvars Hx) | ].
      split; [ exact Hconts | repeat split ].
  - (* A [Let_static]. *)
    rewrite step_let_static,
      (local_bindings_eval M vars vars' bindings Hvars Hbindings statics).
    destruct (eval_static_bindings vars' bindings statics) as [statics1 | ]
      eqn:Hst; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_static, Hst; reflexivity | ].
    exists K. split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ exact Hconts | repeat split ].
  - (* A non-recursive [Let_cont]. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    exists ((k, sh) :: K). split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | repeat split ].
    constructor; [ | exact Hconts ].
    exact (cont_rel_closure M sh ps ps' md md' h h' vars vars' K [] conts
             conts' [] [] Hps Hmd Hmd' Hh Hvars Hconts
             (ub_handlers_nil M K)).
  - (* A recursive [Let_cont]. *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hb; [ exact I | ].
    right. left. eexists.
    split.
    { rewrite step_let_cont_recursive.
      destruct (handlers_bind_statics handlers') eqn:Hb';
        [ congruence | reflexivity ]. }
    exists (S ++ K). split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ apply rec_closures_rel; assumption | repeat split ].
  - (* An [Apply_cont]. *)
    destruct Hac as (Hk & Htrap & Hargs).
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    destruct (resolve_rel M K conts conts' (ac_continuation ac) Hconts)
      as [(HK & Hc & Hc') | (sh & HK & Hc)].
    + unfold shapes_at in Hargs. rewrite HK in Hargs.
      destruct (args_rel_nil_inv _ _ Hargs) as [Heq Hplain].
      rewrite Hc. cbn [apply_cont_value]. exists vs, rcd.
      split; [ | reflexivity ].
      rewrite step_apply_cont, Heq,
        <- (plain_simples_eval M vars vars' _ Hvars Hplain), Hvs.
      unfold apply_continuation. rewrite Hk, Hc'. reflexivity.
    + unfold shapes_at in Hargs. rewrite HK in Hargs.
      remember (resolve (ac_continuation ac) conts) as c eqn:Ec.
      remember (resolve (ac_continuation ac) conts') as c' eqn:Ec'.
      destruct Hc as [sh ps ps' md md' h h' cvars cvars' Kc S cconts cconts'
                        rec rec' Hps Hmd Hmd' Hh Hcv Hcc Hrec].
      cbn [apply_cont_value]. rewrite Hmd.
      destruct (kinds_eqb (map kind_of_value vs) (map snd ps)) eqn:Hkinds;
        [ | exact I ].
      apply FlambdaSemanticsProofs.kinds_eqb_eq in Hkinds.
      destruct (bind_params (map fst ps) vs cvars) as [hv | ] eqn:Hbind;
        [ | exact I ].
      destruct (bind_args_rel sh _ _ Hargs ps ps' vars vars' cvars cvars' hv vs
                  Hps Hvars Hcv Hvs Hkinds Hbind)
        as (vs' & hv' & Hvs' & Hk' & Hb' & Hrel).
      right. left. eexists. split.
      * rewrite step_apply_cont, Hvs'. unfold apply_continuation.
        rewrite Hk, <- Ec'. cbn [apply_cont_value]. rewrite Hmd', <- Hk'.
        rewrite (proj2 (FlambdaSemanticsProofs.kinds_eqb_eq _ _) eq_refl), Hb'.
        reflexivity.
      * exists (S ++ Kc). split; [ exact Hh | split; [ exact Hrel | ] ].
        split; [ apply rec_closures_rel; assumption | repeat split ].
  - (* A [Switch]. *)
    rewrite step_switch,
      (plain_simple_eval M vars vars' scrutinee Hvars Hscrutinee).
    destruct (eval_simple vars' scrutinee) as [w | ] eqn:Hw; [ | exact I ].
    destruct (naked_immediate_of_value w) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    destruct (find_arm_rel K i arms arms' Harms ac Harm) as (ac' & Harm' & Hac).
    right. left. eexists.
    split; [ rewrite step_switch, Hw, Hi, Harm'; reflexivity | ].
    exists K. split; [ constructor; exact Hac | ].
    split; [ exact Hvars | split; [ exact Hconts | repeat split ] ].
  - rewrite step_invalid. exact I.
Qed.

(* Related terms, run on their own, refine each other. *)
Lemma ub_refines (p q : expr) :
  ub M [] p q ->
  refines (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hub. apply (shrinking_refines (related M) eq measure).
  - exact related_answered.
  - intros vs vs' ->. reflexivity.
  - exists []. split; [ exact Hub | split; [ apply env_rel_empty | ] ].
    split; [ constructor | repeat split ].
Qed.

End Simulation.

(* ======================================================================== *)
(* Where the unboxed fields come from                                       *)
(* ======================================================================== *)

Section Provenance.

Variable F : facts.
Variable T : tables.
Variable code_my_closures : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

(* The graph reads only fields of blocks and of boxed numbers. *)
Hypothesis reads_blocks :
  forallb (fun '(_, f, _) => is_block_like_field f) (f_accessor F) = true.

Lemma insert_field_in {A : Type} (f : field) (a : A)
    (l : list (field * A)) (x : field * A) :
  x ∈ insert_field f a l -> x = (f, a) ∨ x ∈ l.
Proof.
  induction l as [ | [f' a'] l IH]; simpl;
    [ intros [<- | []]; left; reflexivity | ].
  destruct (field_lt f f'); simpl.
  - intros [<- | Hin]; [ left; reflexivity | right; exact Hin ].
  - intros [<- | Hin]; [ right; left; reflexivity | ].
    destruct (IH Hin) as [-> | Hin']; [ left; reflexivity | right; right ].
    exact Hin'.
Qed.

Lemma sort_fields_in {A : Type} (l : list (field * A)) (x : field * A) :
  x ∈ sort_fields l -> x ∈ l.
Proof.
  unfold sort_fields. induction l as [ | [f a] l IH]; simpl; [ easy | ].
  intros Hin. apply insert_field_in in Hin as [-> | Hin];
    [ left; reflexivity | right; exact (IH Hin) ].
Qed.

(* With no read of a function slot or a call witness, the usages of the
   closures add nothing. *)
Lemma add_usages_none (follow : bool) (out : list node) :
  add_usages_through_function_slots F T code_my_closures follow out = out.
Proof.
  assert (Hnone :
    usages_through_function_slots F T code_my_closures follow out = []).
  { unfold usages_through_function_slots.
    induction out as [ | x out IH]; [ reflexivity | ]. simpl. rewrite IH.
    rewrite app_nil_r. rewrite forallb_forall in reads_blocks.
    assert (Hacc : ∀ l, (∀ a, a ∈ l -> a ∈ f_accessor F) ->
      (if follow
       then
         flat_map
           (fun '(apply_witness, f, base) =>
              if node_eqb base x
                 && field_eqb f (Field_call_witness Known_arity_code_pointer)
              then
                flat_map
                  (fun '(apply_witness', call_witness) =>
                     if node_eqb apply_witness apply_witness'
                     then
                       flat_map
                         (fun '(call_witness', f', code_id) =>
                            if node_eqb call_witness call_witness'
                               && field_eqb f' Field_code_id_of_call_witness
                            then
                              match code_id with
                              | Node_code_id id =>
                                match lookup id code_my_closures with
                                | Some my_closure =>
                                  flat_map
                                    (fun '(my_closure', y) =>
                                       if node_eqb (Node_variable my_closure)
                                            my_closure'
                                          && mem_node y (t_has_usage T)
                                       then [y]
                                       else [])
                                    (t_usages T)
                                | None => []
                                end
                              | _ => []
                              end
                            else [])
                         (f_constructor F)
                     else [])
                  (t_sources T)
              else [])
           l
       else [])
      ++ flat_map
           (fun '(y, f, base) =>
              if node_eqb base x && is_function_slot f
              then
                flat_map
                  (fun '(y', z) =>
                     if node_eqb y y' && mem_node z (t_has_usage T) then [z]
                     else [])
                  (t_usages T)
              else [])
           l
      = []).
    { induction l as [ | [[y f] base] l IHl]; intros Hl;
        [ destruct follow; reflexivity | ].
      pose proof (reads_blocks _ (Hl _ (or_introl eq_refl))) as Hf.
      cbn beta iota in Hf.
      assert (Hrest : ∀ a, a ∈ l -> a ∈ f_accessor F)
        by (intros a Ha; apply Hl; right; exact Ha).
      specialize (IHl Hrest).
      destruct f; try discriminate Hf; destruct follow; simpl;
        rewrite ?andb_false_r; simpl; exact IHl. }
    exact (Hacc (f_accessor F) (fun a Ha => Ha)). }
  unfold add_usages_through_function_slots. simpl. rewrite Hnone.
  reflexivity.
Qed.

Lemma dedup_in (l : list node) (x : node) :
  x ∈ Keyed.dedup node_eqb l -> x ∈ l.
Proof.
  rewrite <- !(keyed_among_in node node_eqb node_eqb_decides).
  rewrite (keyed_among_dedup node node_eqb node_eqb_decides).
  exact (fun H => H).
Qed.

Lemma get_direct_usages_in (x base : node) :
  base ∈ get_direct_usages T [x] ->
  mem_pair x base (t_usages T) = true ∧ mem_node base (t_has_usage T) = true.
Proof.
  unfold get_direct_usages. intros Hin.
  apply dedup_in in Hin.
  apply in_flat_map in Hin as ([x' y] & Hin & Hy).
  simpl in Hy. rewrite orb_false_r in Hy.
  destruct (node_eqb x' x && mem_node y (t_has_usage T)) eqn:Htest;
    [ | destruct Hy ].
  destruct Hy as [<- | []]. apply andb_true_iff in Htest as [Hx Hy].
  apply node_eqb_eq in Hx. subst x'.
  split; [ apply mem_pair_In; exact Hin | exact Hy ].
Qed.

(* A field of [get_fields] is read, from one of the nodes, into a node with
   a use. *)
Lemma get_fields_in (uses : list node) (f : field) (use : field_use) :
  (f, use) ∈ get_fields F T uses ->
  ∃ w base : node,
    (w, f, base) ∈ f_accessor F ∧ mem_node base uses = true
    ∧ (mem_node w (t_any_usage T) = true
       ∨ mem_node w (t_has_usage T) = true).
Proof.
  unfold get_fields. intros Hin. apply sort_fields_in in Hin.
  apply in_flat_map in Hin as (f0 & _ & Hin).
  set (reads := filter
                  (fun '(_, f, base) =>
                     mem_node base uses && negb (is_function_slot f))
                  (f_accessor F)) in Hin.
  assert (Hreads : ∀ w, w ∈ flat_map
                       (fun '(v, f', _) => if field_eqb f0 f' then [v] else [])
                       reads ->
                     ∃ base, (w, f0, base) ∈ f_accessor F
                             ∧ mem_node base uses = true).
  { intros w Hw. apply in_flat_map in Hw as ([[v f'] base] & Hr & Hv).
    destruct (field_eqb f0 f') eqn:Hf; [ | destruct Hv ].
    destruct Hv as [<- | []]. apply field_eqb_eq in Hf. subst f'.
    unfold reads in Hr. apply filter_In in Hr as [Hr Htest].
    apply andb_true_iff in Htest as [Hbase _].
    exists base. split; assumption. }
  destruct (existsb (fun v => mem_node v (t_any_usage T)) _) eqn:Hany.
  - destruct Hin as [[= <- _] | []].
    apply existsb_exists in Hany as (w & Hw & Hau).
    destruct (Hreads w Hw) as (base & Hacc & Hbase).
    exists w, base. split; [ exact Hacc | split; [ exact Hbase | left ] ].
    exact Hau.
  - destruct (Keyed.dedup node_eqb _) as [ | w0 l] eqn:Hdedup;
      [ destruct Hin | ].
    destruct Hin as [[= <- _] | []].
    assert (Hw0 : w0 ∈ Keyed.dedup node_eqb
                    (filter (fun v => mem_node v (t_has_usage T))
                       (flat_map
                          (fun '(v, f', _) =>
                             if field_eqb f0 f' then [v] else [])
                          reads)))
      by (rewrite Hdedup; left; reflexivity).
    apply dedup_in in Hw0.
    apply filter_In in Hw0 as [Hw0 Hhu].
    destruct (Hreads w0 Hw0) as (base & Hacc & Hbase).
    exists w0, base. split; [ exact Hacc | split; [ exact Hbase | right ] ].
    exact Hhu.
Qed.

(* An unboxed field of an allocation is one of the fields given. *)
Lemma mk_unboxed_fields_in (fuel : nat) (block : node) :
  ∀ (fields : list (field * field_use)) (next : positive) (f : field)
    (u : unboxed_field),
    (f, u) ∈ fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                    fuel block fields next) ->
    ∃ use : field_use, (f, use) ∈ fields.
Proof.
  destruct fuel as [ | fuel]; cbn [mk_unboxed_fields];
    (induction fields as [ | [f' use] fields IH]; intros next f u Hin;
     [ destruct Hin | ]);
    match type of Hin with
    | context [let '(_, _) := ?c in _] => destruct c as [u0 next0]
    end;
    match type of Hin with
    | context [let '(_, _) := ?c in _] => destruct c as [rest next1] eqn:Hrest
    end;
    simpl in Hin;
    (destruct u0 as [u0 | ];
     [ destruct Hin as [[= <- _] | Hin];
       [ exists use; left; reflexivity | ] | ]);
    (destruct (IH next0 f u) as (use' & Huse');
     [ rewrite Hrest; exact Hin | exists use'; right; exact Huse' ]).
Qed.

Lemma unboxed_fields_of_names_in (next : positive) (x : node)
    (fields : unboxed_fields) :
  (x, fields)
    ∈ fst (unboxed_fields_of_names F T code_my_closures dominated to_unbox
             next) ->
  ∃ (a : node) (next' : positive),
    fields
    = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
             (S (length to_unbox)) a
             (get_fields F T
                (add_usages_through_function_slots F T code_my_closures true
                   (get_direct_usages T [x])))
             next').
Proof.
  unfold unboxed_fields_of_names.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc : list (node * unboxed_fields)) (n : positive),
    (∀ y fs, (y, fs) ∈ acc ->
       ∃ a next',
         fs = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                     (S (length to_unbox)) a
                     (get_fields F T
                        (add_usages_through_function_slots F T
                           code_my_closures true (get_direct_usages T [y])))
                     next')) ->
    ∀ y fs,
      (y, fs) ∈ fst (List.fold_left
                       (fun '(acc, next) '(a, x) =>
                          if mem_node a to_unbox
                             && negb (existsb (fun '(y, _) => node_eqb x y)
                                        acc)
                          then
                            let '(fields, next) :=
                              name_unboxed_fields F T code_my_closures
                                dominated to_unbox a x next
                            in
                            (acc ++ [(x, fields)], next)
                          else (acc, next))
                       l (acc, n)) ->
      ∃ a next',
        fs = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                    (S (length to_unbox)) a
                    (get_fields F T
                       (add_usages_through_function_slots F T
                          code_my_closures true (get_direct_usages T [y])))
                    next')).
  { induction l as [ | [a x'] l IH]; intros acc n Hacc; [ exact Hacc | ].
    simpl. destruct (mem_node a to_unbox && _); [ | apply IH, Hacc ].
    unfold name_unboxed_fields.
    destruct (mk_unboxed_fields _ _ _ _ _ _ _ _ n) as [fs0 n0] eqn:Hmk.
    apply IH. intros y fs Hin. apply in_app_or in Hin as [Hin | [Heq | []]].
    - exact (Hacc y fs Hin).
    - injection Heq as <- <-. exists a, n. rewrite Hmk. reflexivity. }
  apply (Hgen dominated [] next). intros y fs [].
Qed.

(* The converses, for the reads that the rebuild keeps. *)
Lemma insert_field_in_r {A : Type} (f : field) (a : A)
    (l : list (field * A)) (x : field * A) :
  x = (f, a) ∨ x ∈ l -> x ∈ insert_field f a l.
Proof.
  induction l as [ | [f' a'] l IH]; simpl;
    [ intros [-> | []]; left; reflexivity | ].
  destruct (field_lt f f'); simpl.
  - intros [-> | Hin]; [ left; reflexivity | right; exact Hin ].
  - intros [-> | [-> | Hin]]; [ right; apply IH; left; reflexivity
                              | left; reflexivity
                              | right; apply IH; right; exact Hin ].
Qed.

Lemma sort_fields_in_r {A : Type} (l : list (field * A)) (x : field * A) :
  x ∈ l -> x ∈ sort_fields l.
Proof.
  unfold sort_fields. induction l as [ | [f a] l IH]; simpl; [ easy | ].
  intros [<- | Hin]; apply insert_field_in_r;
    [ left; reflexivity | right; exact (IH Hin) ].
Qed.

Lemma dedup_in_r (l : list node) (x : node) :
  x ∈ l -> x ∈ Keyed.dedup node_eqb l.
Proof.
  rewrite <- !(keyed_among_in node node_eqb node_eqb_decides).
  rewrite (keyed_among_dedup node node_eqb node_eqb_decides).
  exact (fun H => H).
Qed.

Lemma get_direct_usages_self (x : node) :
  mem_pair x x (t_usages T) = true -> mem_node x (t_has_usage T) = true ->
  x ∈ get_direct_usages T [x].
Proof.
  intros Hus Hhu. unfold get_direct_usages. apply dedup_in_r.
  apply in_flat_map. exists (x, x).
  split; [ apply mem_pair_In; exact Hus | ].
  simpl. rewrite node_eqb_refl, Hhu. left. reflexivity.
Qed.

(* A field read off one of [uses] into a node [w] with a use is among
   the fields of [get_fields], and each time with [w] among the nodes it
   flows to, if they are known. *)
Lemma get_fields_found (uses : list node) (w base : node) (f : field) :
  (w, f, base) ∈ f_accessor F -> mem_node base uses = true ->
  is_function_slot f = false -> mem_node w (t_has_usage T) = true ->
  (∃ use : field_use, (f, use) ∈ get_fields F T uses)
  ∧ ∀ use : field_use,
      (f, use) ∈ get_fields F T uses ->
      use = Use_unknown
      ∨ ∃ flow_to : list node, use = Use_known flow_to ∧ w ∈ flow_to.
Proof.
  intros Hacc Hbase Hfs Hw. unfold get_fields.
  set (reads := filter
                  (fun '(_, f, base) =>
                     mem_node base uses && negb (is_function_slot f))
                  (f_accessor F)).
  assert (Hr : (w, f, base) ∈ reads).
  { unfold reads. apply filter_In. split; [ exact Hacc | ].
    rewrite Hbase, Hfs. reflexivity. }
  set (read_into := flat_map
                      (fun '(v, f', _) => if field_eqb f f' then [v] else [])
                      reads).
  assert (Hw_in : w ∈ read_into).
  { unfold read_into. apply in_flat_map. exists (w, f, base).
    split; [ exact Hr | ]. rewrite field_eqb_refl. left. reflexivity. }
  assert (Hw_flow : w ∈ Keyed.dedup node_eqb
                          (filter (fun v => mem_node v (t_has_usage T))
                             read_into)).
  { apply dedup_in_r. apply filter_In. split; [ exact Hw_in | exact Hw ]. }
  split.
  - destruct (existsb (fun v => mem_node v (t_any_usage T)) read_into)
      eqn:Hany.
    + exists Use_unknown. apply sort_fields_in_r. apply in_flat_map.
      exists f. split.
      * apply (keyed_among_in field field_eqb field_eqb_decides).
        rewrite (keyed_among_dedup field field_eqb field_eqb_decides).
        apply (keyed_among_in field field_eqb field_eqb_decides).
        apply in_map_iff. exists (w, f, base). split; [ reflexivity | ].
        exact Hr.
      * fold read_into. rewrite Hany. left. reflexivity.
    + destruct (Keyed.dedup node_eqb
                  (filter (fun v => mem_node v (t_has_usage T)) read_into))
        as [ | w0 l] eqn:Hdedup; [ destruct Hw_flow | ].
      exists (Use_known (w0 :: l)). apply sort_fields_in_r.
      apply in_flat_map. exists f. split.
      * apply (keyed_among_in field field_eqb field_eqb_decides).
        rewrite (keyed_among_dedup field field_eqb field_eqb_decides).
        apply (keyed_among_in field field_eqb field_eqb_decides).
        apply in_map_iff. exists (w, f, base). split; [ reflexivity | ].
        exact Hr.
      * fold read_into. rewrite Hany, Hdedup. left. reflexivity.
  - intros use Hin. apply sort_fields_in in Hin.
    apply in_flat_map in Hin as (f0 & _ & Hin).
    destruct (existsb (fun v => mem_node v (t_any_usage T))
                (flat_map
                   (fun '(v, f', _) => if field_eqb f0 f' then [v] else [])
                   reads)) eqn:Hany.
    + destruct Hin as [[= -> <-] | []]. left. reflexivity.
    + destruct (Keyed.dedup node_eqb
                  (filter (fun v => mem_node v (t_has_usage T))
                     (flat_map
                        (fun '(v, f', _) =>
                           if field_eqb f0 f' then [v] else [])
                        reads)))
        as [ | w0 l] eqn:Hdedup; [ destruct Hin | ].
      destruct Hin as [[= -> <-] | []]. right. exists (w0 :: l).
      split; [ reflexivity | ]. fold read_into in Hdedup.
      rewrite <- Hdedup. exact Hw_flow.
Qed.

(* A field given to [mk_unboxed_fields] that has a source, and that is
   not read into names all to be unboxed, is a variable. *)
Lemma mk_unboxed_fields_lookup (fuel : nat) (block : node) (f : field) :
  is_data_field f = true ->
  get_single_field_source F T block f <> No_source ->
  ∀ (fields : list (field * field_use)) (next : positive),
    (∃ use : field_use, (f, use) ∈ fields) ->
    (∀ use : field_use,
       (f, use) ∈ fields ->
       use = Use_unknown
       ∨ ∃ flow_to : list node,
           use = Use_known flow_to
           ∧ forallb (has_to_be_unboxed dominated to_unbox) flow_to
             = false) ->
    ∃ v : variable,
      Keyed.lookup field_eqb f
        (fst (mk_unboxed_fields F T code_my_closures dominated to_unbox fuel
                block fields next))
      = Some (Not_unboxed v).
Proof.
  intros Hdata Hsource.
  destruct fuel as [ | fuel]; cbn [mk_unboxed_fields];
    (induction fields as [ | [f' use] fields IH]; intros next Hex Hall;
     [ destruct Hex as (use & []) | ]);
    (destruct (field_eq_dec f' f) as [-> | Hne]).
  1, 3:
    destruct (Hall use (or_introl eq_refl)) as [-> | (flow_to & -> & Hft)];
    rewrite Hdata; simpl;
    destruct (get_single_field_source F T block f) eqn:Hs;
    try contradiction;
    (try rewrite Hft);
    match goal with
    | |- context [let '(_, _) := ?c in _] => destruct c as [rest next1]
    end;
    simpl; rewrite field_eqb_refl; eexists; reflexivity.
  all:
    (assert (Hex' : ∃ use0, (f, use0) ∈ fields)
       by (destruct Hex as (use0 & [[= <- _] | Hin]);
           [ contradiction | exists use0; exact Hin ]));
    (assert (Hall' : ∀ use0, (f, use0) ∈ fields ->
              use0 = Use_unknown
              ∨ ∃ flow_to, use0 = Use_known flow_to
                           ∧ forallb (has_to_be_unboxed dominated to_unbox)
                               flow_to = false)
       by (intros use0 Hin; apply Hall; right; exact Hin));
    match goal with
    | |- context [let '(_, _) := ?c in _] => destruct c as [u0 next0]
    end;
    match goal with
    | |- context [let '(_, _) := ?c in _] =>
      destruct c as [rest next1] eqn:Hrest
    end;
    destruct (IH next0 Hex' Hall') as (v & Hv);
    rewrite Hrest in Hv; simpl in Hv; exists v;
    destruct u0 as [u0 | ]; simpl; [ | exact Hv ];
    destruct (field_eqb f f') eqn:Hff;
    [ apply field_eqb_eq in Hff; congruence | exact Hv ].
Qed.

(* The unboxed fields of a name, from an allocation to unbox that
   dominates it. *)
Lemma unboxed_fields_of_names_dom (next : positive) (x : node)
    (fields : unboxed_fields) :
  (x, fields)
    ∈ fst (unboxed_fields_of_names F T code_my_closures dominated to_unbox
             next) ->
  ∃ (a : node) (next' : positive),
    (a, x) ∈ dominated ∧ mem_node a to_unbox = true
    ∧ fields
      = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
               (S (length to_unbox)) a
               (get_fields F T
                  (add_usages_through_function_slots F T code_my_closures true
                     (get_direct_usages T [x])))
               next').
Proof.
  unfold unboxed_fields_of_names.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc : list (node * unboxed_fields)) (n : positive),
    (∀ p, p ∈ l -> p ∈ dominated) ->
    (∀ y fs, (y, fs) ∈ acc ->
       ∃ a next',
         (a, y) ∈ dominated ∧ mem_node a to_unbox = true
         ∧ fs = fst (mk_unboxed_fields F T code_my_closures dominated
                       to_unbox (S (length to_unbox)) a
                       (get_fields F T
                          (add_usages_through_function_slots F T
                             code_my_closures true (get_direct_usages T [y])))
                       next')) ->
    ∀ y fs,
      (y, fs) ∈ fst (List.fold_left
                       (fun '(acc, next) '(a, x) =>
                          if mem_node a to_unbox
                             && negb (existsb (fun '(y, _) => node_eqb x y)
                                        acc)
                          then
                            let '(fields, next) :=
                              name_unboxed_fields F T code_my_closures
                                dominated to_unbox a x next
                            in
                            (acc ++ [(x, fields)], next)
                          else (acc, next))
                       l (acc, n)) ->
      ∃ a next',
        (a, y) ∈ dominated ∧ mem_node a to_unbox = true
        ∧ fs = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                      (S (length to_unbox)) a
                      (get_fields F T
                         (add_usages_through_function_slots F T
                            code_my_closures true (get_direct_usages T [y])))
                      next')).
  { induction l as [ | [a x'] l IH]; intros acc n Hl Hacc; [ exact Hacc | ].
    simpl. destruct (mem_node a to_unbox) eqn:Ha; simpl;
      [ | apply IH; [ intros p Hp; apply Hl; right; exact Hp | exact Hacc ] ].
    destruct (negb _); [ | apply IH;
                           [ intros p Hp; apply Hl; right; exact Hp
                           | exact Hacc ] ].
    unfold name_unboxed_fields.
    destruct (mk_unboxed_fields _ _ _ _ _ _ _ _ n) as [fs0 n0] eqn:Hmk.
    apply IH; [ intros p Hp; apply Hl; right; exact Hp | ].
    intros y fs Hin. apply in_app_or in Hin as [Hin | [Heq | []]].
    - exact (Hacc y fs Hin).
    - injection Heq as <- <-. exists a, n.
      split; [ apply Hl; left; reflexivity | split; [ exact Ha | ] ].
      rewrite Hmk. reflexivity. }
  apply (Hgen dominated [] next); [ exact (fun p Hp => Hp) | ].
  intros y fs [].
Qed.

(* Each name that an allocation to unbox dominates has unboxed fields. *)
Lemma unboxed_fields_of_names_bound (next : positive) (a x : node) :
  (a, x) ∈ dominated -> mem_node a to_unbox = true ->
  x ∈ map fst (fst (unboxed_fields_of_names F T code_my_closures dominated
                      to_unbox next)).
Proof.
  intros Hin Ha. unfold unboxed_fields_of_names.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc : list (node * unboxed_fields)) (n : positive),
    x ∈ map fst acc ∨ (a, x) ∈ l ->
    x ∈ map fst
          (fst (List.fold_left
                  (fun '(acc, next) '(a, x) =>
                     if mem_node a to_unbox
                        && negb (existsb (fun '(y, _) => node_eqb x y) acc)
                     then
                       let '(fields, next) :=
                         name_unboxed_fields F T code_my_closures dominated
                           to_unbox a x next
                       in
                       (acc ++ [(x, fields)], next)
                     else (acc, next))
                  l (acc, n)))).
  { induction l as [ | [a' x'] l IH]; intros acc n H;
      [ destruct H as [H | []]; exact H | ].
    simpl.
    destruct (mem_node a' to_unbox
              && negb (existsb (fun '(y, _) => node_eqb x' y) acc))
      eqn:Htest.
    - unfold name_unboxed_fields.
      destruct (mk_unboxed_fields _ _ _ _ _ _ _ _ n) as [fs0 n0].
      apply IH. rewrite map_app.
      destruct H as [H | [[= -> ->] | H]];
        [ left; apply in_or_app; left; exact H
        | left; apply in_or_app; right; left; reflexivity
        | right; exact H ].
    - apply IH.
      destruct H as [H | [[= -> ->] | H]]; [ left; exact H | | right; exact H ].
      left. rewrite Ha in Htest. simpl in Htest.
      apply negb_false_iff in Htest. apply existsb_exists in Htest
        as ([y fs] & Hy & Heq). apply node_eqb_eq in Heq. subst y.
      apply in_map_iff. exists (x, fs). split; [ reflexivity | exact Hy ]. }
  apply Hgen. right. exact Hin.
Qed.

End Provenance.

(* ======================================================================== *)
(* The variables of the fields are fresh                                    *)
(* ======================================================================== *)

Section Fresh.

Variable F : facts.
Variable T : tables.
Variable code_my_closures : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

(* The variables of [fields] are distinct, from [lo] on and before
   [hi]. *)
Definition fresh_range (lo hi : positive) (fields : unboxed_fields) : Prop :=
  (lo <= hi)%positive ∧ NoDup (leaf_variables fields)
  ∧ ∀ v, v ∈ leaf_variables fields -> (lo <= v)%positive ∧ (v < hi)%positive.

Lemma leaf_variables_cons_var (f : field) (v : variable)
    (rest : unboxed_fields) :
  leaf_variables ((f, Not_unboxed v) :: rest) = v :: leaf_variables rest.
Proof. reflexivity. Qed.

(* A field, unboxed or not, then the rest. *)
Lemma fresh_range_app (lo mid hi : positive) (u : option unboxed_field)
    (f : field) (rest : unboxed_fields) :
  (match u with
   | Some (Not_unboxed v) => lo = v ∧ mid = Pos.succ v
   | Some (Unboxed nested) => fresh_range lo mid nested
   | None => lo = mid
   end) ->
  fresh_range mid hi rest ->
  fresh_range lo hi
    (match u with Some u => (f, u) :: rest | None => rest end).
Proof.
  intros Hu (Hmh & Hnd & Hrest). unfold fresh_range.
  destruct u as [[v | nested] | ]; cbv iota.
  - destruct Hu as [-> ->]. rewrite leaf_variables_cons_var.
    split; [ lia | split ].
    + constructor; [ | exact Hnd ]. intros Hv.
      destruct (Hrest v Hv) as [H1 _]. lia.
    + intros w [<- | Hw]; [ lia | ]. destruct (Hrest w Hw). lia.
  - destruct Hu as (Hlm & Hnd' & Hnested). rewrite leaf_variables_nested.
    split; [ lia | split ].
    + apply NoDup_app; [ exact Hnd' | exact Hnd | ].
      intros w Hw Hw'. destruct (Hnested w Hw). destruct (Hrest w Hw'). lia.
    + intros w Hw. apply in_app_or in Hw as [Hw | Hw];
        [ destruct (Hnested w Hw) | destruct (Hrest w Hw) ]; lia.
  - subst mid. split; [ exact Hmh | split; [ exact Hnd | exact Hrest ] ].
Qed.

(* One field of [mk_unboxed_fields], as its inner function makes it. *)
Definition field_step (fuel : nat) (block : node) (f : field) (use : field_use)
    (next : positive) : option unboxed_field * positive :=
  if negb (is_data_field f)
  then (None, next)
  else
    match get_single_field_source F T block f with
    | No_source => (None, next)
    | field_source =>
      let default := (Some (Not_unboxed next), Pos.succ next) in
      match use with
      | Use_unknown => default
      | Use_known flow_to =>
        if forallb (has_to_be_unboxed dominated to_unbox) flow_to
        then
          match field_source, fuel with
          | One_source source, S fuel =>
            let '(nested, next) :=
              mk_unboxed_fields F T code_my_closures dominated to_unbox fuel
                source
                (get_fields F T
                   (add_usages_through_function_slots F T code_my_closures
                      true (get_direct_usages T flow_to)))
                next
            in
            (Some (Unboxed nested), next)
          | _, _ => default
          end
        else default
      end
    end.

Lemma mk_unboxed_fields_cons (fuel : nat) (block : node) (f : field)
    (use : field_use) (fields : list (field * field_use)) (next : positive) :
  mk_unboxed_fields F T code_my_closures dominated to_unbox fuel block
    ((f, use) :: fields) next
  = let '(u, next) := field_step fuel block f use next in
    let '(rest, next) :=
      mk_unboxed_fields F T code_my_closures dominated to_unbox fuel block
        fields next
    in
    (match u with
     | Some u => (f, u) :: rest
     | None => rest
     end,
     next).
Proof. destruct fuel; reflexivity. Qed.

Lemma mk_unboxed_fields_nil (fuel : nat) (block : node) (next : positive) :
  mk_unboxed_fields F T code_my_closures dominated to_unbox fuel block [] next
  = ([], next).
Proof. destruct fuel; reflexivity. Qed.

(* [mk_unboxed_fields] names the variables from [next] on, each once. *)
Lemma mk_unboxed_fields_fresh (fuel : nat) :
  ∀ (block : node) (fields : list (field * field_use)) (next : positive),
    let r := mk_unboxed_fields F T code_my_closures dominated to_unbox fuel
               block fields next in
    fresh_range next (snd r) (fst r).
Proof.
  induction fuel as [ | fuel IHfuel]; intros block fields;
    induction fields as [ | [f use] fields IH]; intros next; cbv zeta;
    rewrite ?mk_unboxed_fields_nil, ?mk_unboxed_fields_cons;
    try (cbn [fst snd];
         split; [ lia | split; [ constructor | intros v [] ] ]).
  - assert (Hstep : let '(u, n) := field_step 0 block f use next in
                    match u with
                    | Some (Not_unboxed v) => next = v ∧ n = Pos.succ v
                    | Some (Unboxed nested) => fresh_range next n nested
                    | None => next = n
                    end).
    { unfold field_step. destruct (negb (is_data_field f)); [ reflexivity | ].
      destruct (get_single_field_source F T block f) as [ | source | ];
        [ reflexivity | | ];
        destruct use as [ | flow_to ]; [ split; reflexivity | |
                                         split; reflexivity | ];
        destruct (forallb _ flow_to); split; reflexivity. }
    destruct (field_step 0 block f use next) as [u n].
    specialize (IH n). cbv zeta in IH.
    destruct (mk_unboxed_fields F T code_my_closures dominated to_unbox 0 block
                fields n) as [rest n1].
    exact (fresh_range_app next n n1 u f rest Hstep IH).
  - assert (Hstep : let '(u, n) := field_step (S fuel) block f use next in
                    match u with
                    | Some (Not_unboxed v) => next = v ∧ n = Pos.succ v
                    | Some (Unboxed nested) => fresh_range next n nested
                    | None => next = n
                    end).
    { unfold field_step. destruct (negb (is_data_field f)); [ reflexivity | ].
      destruct (get_single_field_source F T block f) as [ | source | ];
        [ reflexivity | | ];
        destruct use as [ | flow_to ]; [ split; reflexivity | |
                                         split; reflexivity | ];
        destruct (forallb _ flow_to); try (split; reflexivity).
      pose proof (IHfuel source
                    (get_fields F T
                       (add_usages_through_function_slots F T code_my_closures
                          true (get_direct_usages T flow_to)))
                    next) as Hn.
      cbv zeta in Hn.
      destruct (mk_unboxed_fields _ _ _ _ _ fuel source _ next) as [nested n].
      exact Hn. }
    destruct (field_step (S fuel) block f use next) as [u n].
    specialize (IH n). cbv zeta in IH.
    destruct (mk_unboxed_fields F T code_my_closures dominated to_unbox (S fuel)
                block fields n) as [rest n1].
    exact (fresh_range_app next n n1 u f rest Hstep IH).
Qed.

Lemma field_variables_app (l l' : list (node * unboxed_fields)) :
  field_variables (l ++ l') = field_variables l ++ field_variables l'.
Proof. unfold field_variables. apply flat_map_app. Qed.

Lemma field_variables_one (x : node) (fields : unboxed_fields) :
  field_variables [(x, fields)] = leaf_variables fields.
Proof. unfold field_variables. simpl. rewrite app_nil_r. reflexivity. Qed.

(* The variables of the fields of the names are distinct, from [next] on. *)
Lemma unboxed_fields_of_names_fresh (start : positive) :
  let r := unboxed_fields_of_names F T code_my_closures dominated to_unbox
             start in
  (start <= snd r)%positive ∧ NoDup (field_variables (fst r))
  ∧ ∀ v, v ∈ field_variables (fst r) ->
      (start <= v)%positive ∧ (v < snd r)%positive.
Proof.
  unfold unboxed_fields_of_names.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc : list (node * unboxed_fields)) (n : positive),
    (start <= n)%positive -> NoDup (field_variables acc) ->
    (∀ v, v ∈ field_variables acc ->
       (start <= v)%positive ∧ (v < n)%positive) ->
    let r := List.fold_left
               (fun '(acc, next) '(a, x) =>
                  if mem_node a to_unbox
                     && negb (existsb (fun '(y, _) => node_eqb x y) acc)
                  then
                    let '(fields, next) :=
                      name_unboxed_fields F T code_my_closures dominated
                        to_unbox a x next
                    in
                    (acc ++ [(x, fields)], next)
                  else (acc, next))
               l (acc, n)
    in
    (start <= snd r)%positive ∧ NoDup (field_variables (fst r))
    ∧ ∀ v, v ∈ field_variables (fst r) ->
        (start <= v)%positive ∧ (v < snd r)%positive).
  { induction l as [ | [a x] l IH]; intros acc n Hn Hnd Hin;
      [ split; [ exact Hn | split; [ exact Hnd | exact Hin ] ] | ].
    simpl. destruct (mem_node a to_unbox && _); [ | apply IH; assumption ].
    unfold name_unboxed_fields.
    pose proof (mk_unboxed_fields_fresh (S (length to_unbox)) a
                  (get_fields F T
                     (add_usages_through_function_slots F T code_my_closures
                        true (get_direct_usages T [x])))
                  n) as Hfresh.
    cbv zeta in Hfresh.
    destruct (mk_unboxed_fields _ _ _ _ _ (S (length to_unbox)) a _ n)
      as [fields n'].
    destruct Hfresh as (Hnn' & Hnd' & Hrange). cbn [fst snd] in *.
    apply IH.
    - lia.
    - rewrite field_variables_app, field_variables_one.
      apply NoDup_app; [ exact Hnd | exact Hnd' | ].
      intros v Hv Hv'. destruct (Hin v Hv). destruct (Hrange v Hv'). lia.
    - intros v Hv. rewrite field_variables_app, field_variables_one in Hv.
      apply in_app_or in Hv as [Hv | Hv];
        [ destruct (Hin v Hv) | destruct (Hrange v Hv) ]; lia. }
  apply Hgen; [ lia | constructor | intros v [] ].
Qed.

Lemma fold_max_le (l : list positive) (z : positive) :
  z ∈ l -> (z <= fold_right Pos.max 1 l)%positive.
Proof.
  induction l as [ | a l IH]; intros Hz; [ destruct Hz | ].
  simpl. destruct Hz as [<- | Hz]; [ lia | specialize (IH Hz); lia ].
Qed.

End Fresh.

(* The variables of the fields that the tables give are distinct, and are
   no variable of the unit. *)
Lemma unit_unboxed_fields_fresh (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr) :
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  distinct (field_variables unboxed) = true
  ∧ ∀ z : variable,
      z ∈ variables_of e -> among z (field_variables unboxed) = false.
Proof.
  unfold unit_unboxed_fields. cbv zeta.
  destruct (unboxed_fields_of_names_fresh F T (bound_code_my_closures e)
              dominated to_unbox (fresh_name (names_of e)))
    as (_ & Hnd & Hrange).
  split; [ apply distinct_NoDup, Hnd | ].
  intros z Hz. destruct (among z _) eqn:Hamong; [ | reflexivity ].
  apply among_In in Hamong. destruct (Hrange z Hamong) as [Hle _].
  assert (Hnames : z ∈ names_of e)
    by (unfold names_of; apply in_or_app; left; exact Hz).
  apply fold_max_le in Hnames. unfold fresh_name in Hle. lia.
Qed.


(* The checks of FlambdaReaperUnboxingRefinesSpec.v, with the variables of
   the unit fresh, are those above. *)
Section Strong.

Variable unboxed : list (node * unboxed_fields).
Variable sol : solution.

Abbreviation fresh z := (among z (field_variables unboxed) = false).

Lemma strong_variable (x : variable) :
  FlambdaReaperUnboxingRefinesSpec.plain_variable unboxed x = true ->
  fresh x -> plain_variable unboxed x = true.
Proof.
  unfold plain_variable, FlambdaReaperUnboxingRefinesSpec.plain_variable.
  intros -> ->. reflexivity.
Qed.

Lemma strong_simple (s : simple) :
  FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed s = true ->
  (∀ z, z ∈ variables_in_simple s -> fresh z) ->
  plain_simple unboxed s = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs Hz;
    [ | exact Hs | reflexivity ].
  apply strong_variable; [ exact Hs | apply Hz; left; reflexivity ].
Qed.

Lemma strong_simples (l : list simple) :
  forallb (FlambdaReaperUnboxingRefinesSpec.plain_simple unboxed) l = true ->
  (∀ z, z ∈ variables_in_simples l -> fresh z) ->
  forallb (plain_simple unboxed) l = true.
Proof.
  induction l as [ | s l IH]; simpl; [ reflexivity | ].
  rewrite !andb_true_iff. intros [Hs Hl] Hz.
  unfold variables_in_simples in Hz. simpl in Hz.
  split.
  - apply strong_simple; [ exact Hs | ].
    intros z Hin. apply Hz, in_or_app. left. exact Hin.
  - apply IH; [ exact Hl | ]. intros z Hin. apply Hz, in_or_app. right.
    exact Hin.
Qed.

Lemma strong_unboxed_binding (kinds : finite_map flambda_kind)
    (x : variable) (kind : flambda_kind) (d : named)
    (fields : unboxed_fields) :
  FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok unboxed sol kinds x kind
    d fields
  = true ->
  (∀ z, z ∈ variables_in_named d -> fresh z) ->
  unboxed_binding_ok unboxed sol kinds x kind d fields = true.
Proof.
  unfold unboxed_binding_ok,
    FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok.
  intros Hd Hz. destruct kind; try discriminate Hd.
  destruct d as [s | p args | | | | | ]; try discriminate Hd;
    [ destruct s; try discriminate Hd; exact Hd | ].
  destruct p as [p | p | p | p | p]; try discriminate Hd.
  - destruct p; try discriminate Hd;
      destruct args as [ | c [ | ]]; try discriminate Hd; try exact Hd.
    apply andb_true_iff in Hd as [Hc Hfields].
    rewrite (strong_simple c Hc); [ exact Hfields | ].
    intros z Hin. apply Hz. simpl. unfold variables_in_simples. simpl.
    rewrite app_nil_r. exact Hin.
  - destruct p as [[tag] | ]; try discriminate Hd.
    apply forallb_forall. intros [f u] Hin. rewrite forallb_forall in Hd.
    specialize (Hd _ Hin). cbn beta iota in Hd |- *.
    destruct f as [i k | | | | | | | | ]; try discriminate Hd.
    destruct k; try discriminate Hd.
    destruct u as [v | nested]; [ | exact Hd ].
    destruct (nth_error args i) as [a | ] eqn:Hnth; [ | discriminate Hd ].
    apply andb_true_iff in Hd as [Ha Hk]. rewrite Hk, andb_true_r.
    apply (strong_simple a Ha). intros z Hza. apply Hz. simpl.
    apply nth_error_In in Hnth. clear -Hnth Hza.
    induction args as [ | a' args IH]; [ destruct Hnth | ].
    unfold variables_in_simples. simpl. apply in_or_app.
    destruct Hnth as [<- | Hnth]; [ left; exact Hza | right ].
    exact (IH Hnth).
Qed.

Lemma strong_plain_binding (x : variable) (kind : flambda_kind) (d : named) :
  FlambdaReaperUnboxingRefinesSpec.plain_binding_ok unboxed x kind d
  = true ->
  (∀ z, z ∈ variables_in_named d -> fresh z) ->
  plain_binding_ok unboxed x kind d = true.
Proof.
  unfold plain_binding_ok, FlambdaReaperUnboxingRefinesSpec.plain_binding_ok.
  destruct (unboxed_read unboxed d) as [[f fields] | ]; [ easy | ].
  intros Hd Hz. destruct d as [s | p args | | | | | ]; try discriminate Hd.
  - exact (strong_simple s Hd Hz).
  - exact (strong_simples args Hd Hz).
Qed.

Lemma strong_args_ok (kinds : finite_map flambda_kind) (args : list simple) :
  ∀ params : list (variable * bool),
    FlambdaReaperUnboxingRefinesSpec.args_ok unboxed sol kinds params args
    = true ->
    (∀ z, z ∈ variables_in_simples args -> fresh z) ->
    args_ok unboxed sol kinds params args = true.
Proof.
  induction args as [ | a args IH]; intros params Hok Hz;
    [ destruct params as [ | [? ?] ?]; reflexivity | ].
  unfold variables_in_simples in Hz. simpl in Hz.
  assert (Ha : ∀ z, z ∈ variables_in_simple a -> fresh z)
    by (intros z Hin; apply Hz, in_or_app; left; exact Hin).
  assert (Hargs : ∀ z, z ∈ variables_in_simples args -> fresh z)
    by (intros z Hin; apply Hz, in_or_app; right; exact Hin).
  destruct params as [ | [p keep] params]; simpl in Hok |- *.
  - apply andb_true_iff in Hok as [Ha' Hargs'].
    rewrite (strong_simple a Ha' Ha).
    exact (strong_simples args Hargs' Hargs).
  - apply andb_true_iff in Hok as [Hp Hrest].
    rewrite (IH params Hrest Hargs), andb_true_r.
    destruct (fields_of unboxed p); [ exact Hp | ].
    destruct keep; [ | reflexivity ]. exact (strong_simple a Hp Ha).
Qed.

Lemma strong_apply_cont_ok (kinds : finite_map flambda_kind)
    (conts : finite_map (list (variable * bool))) (ac : apply_cont) :
  FlambdaReaperUnboxingRefinesSpec.apply_cont_ok unboxed sol kinds conts ac
  = true ->
  (∀ z, z ∈ variables_in_simples (ac_args ac) -> fresh z) ->
  apply_cont_ok unboxed sol kinds conts ac = true.
Proof.
  unfold apply_cont_ok, FlambdaReaperUnboxingRefinesSpec.apply_cont_ok.
  intros Hok Hz. destruct (lookup (ac_continuation ac) conts).
  - exact (strong_args_ok kinds _ _ Hok Hz).
  - exact (strong_simples _ Hok Hz).
Qed.

Lemma strong_local :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (conts : finite_map (list (variable * bool))),
     FlambdaReaperUnboxingRefinesSpec.local_expr unboxed sol kinds conts e
     = true ->
     (∀ z, z ∈ variables_of e -> fresh z) ->
     local_expr unboxed sol kinds conts e = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (conts : finite_map (list (variable * bool))) (mask : list bool),
       FlambdaReaperUnboxingRefinesSpec.local_handler unboxed sol kinds conts
         mask h
       = true ->
       (∀ z, z ∈ variables_in_handler h -> fresh z) ->
       local_handler unboxed sol kinds conts mask h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hd Hbody].
    rewrite (Hz x (or_introl eq_refl)). simpl.
    apply andb_true_iff. split.
    + destruct (fields_of unboxed x) as [fields | ].
      * apply (strong_unboxed_binding _ _ _ _ _ Hd).
        intros z Hin. apply Hz. right. apply in_or_app. left. exact Hin.
      * apply (strong_plain_binding _ _ _ Hd).
        intros z Hin. apply Hz. right. apply in_or_app. left. exact Hin.
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. right. apply in_or_app.
      right. exact Hin.
  - intros bindings body _ IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    apply andb_true_iff. split.
    + apply forallb_forall. intros b Hb. rewrite forallb_forall in Hbs.
      specialize (Hbs b Hb).
      destruct b as [id c | id | sym [ | | tag fields | | | ]];
        try discriminate Hbs.
      apply (strong_simples fields Hbs). intros z Hin. apply Hz.
      apply in_or_app. left. apply in_flat_map. exists (Bind_symbol sym
        (Static_const_block tag fields)). split; [ exact Hb | exact Hin ].
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros k h body IHh IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hh Hbody].
    apply andb_true_iff. split.
    + apply (IHh _ _ _ Hh). intros z Hin. apply Hz. apply in_or_app. left.
      exact Hin.
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros handlers body IHh IH kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    apply andb_true_iff. split.
    + apply forallb_forall. intros [k h] Hin. rewrite forallb_forall in Hhs.
      apply (IHh k h Hin _ _ _ (Hhs _ Hin)). intros z Hz'. apply Hz.
      apply in_or_app. left. apply in_flat_map. exists (k, h).
      split; [ exact Hin | exact Hz' ].
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
  - intros ap kinds conts Hloc. discriminate Hloc.
  - intros ac kinds conts Hloc Hz. simpl in Hloc |- *.
    exact (strong_apply_cont_ok _ _ _ Hloc Hz).
  - intros scrutinee arms kinds conts Hloc Hz. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hs Harms].
    apply andb_true_iff. split.
    + apply (strong_simple _ Hs). intros z Hin. apply Hz. apply in_or_app.
      left. exact Hin.
    + apply forallb_forall. intros [d ac] Hin. rewrite forallb_forall in Harms.
      apply (strong_apply_cont_ok _ _ _ (Harms _ Hin)). intros z Hz'.
      apply Hz. apply in_or_app. right. apply in_flat_map. exists (d, ac).
      split; [ exact Hin | exact Hz' ].
  - intros kinds conts _ _. reflexivity.
  - intros params metadata handler IH kinds conts mask Hloc Hz.
    simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hlen Hparams].
    rewrite Hlen. simpl. apply andb_true_iff. split.
    + apply forallb_forall. intros [[p kind] keep] Hp.
      rewrite forallb_forall in Hparams. specialize (Hparams _ Hp).
      unfold param_ok. rewrite Hparams, andb_true_r.
      apply in_combine_l, in_combine_l in Hp.
      apply negb_true_iff. apply Hz. apply in_or_app. left. exact Hp.
    + apply (IH _ _ Hbody). intros z Hin. apply Hz. apply in_or_app. right.
      exact Hin.
Qed.

End Strong.

(* ======================================================================== *)
(* The reads of the graph of a unit without closures                       *)
(* ======================================================================== *)

(* The accessor facts of a list read fields of blocks and of boxed numbers
   only. *)
Definition reads_blocks_only (l : list fact) : Prop :=
  ∀ (to_ : node) (f : field) (base : node),
    Fact_accessor to_ f base ∈ l -> is_block_like_field f = true.

Abbreviation rb a := (reads_blocks_only (acc_facts a)).

Lemma rb_app (l l' : list fact) :
  reads_blocks_only l -> reads_blocks_only l' -> reads_blocks_only (l ++ l').
Proof.
  intros H H' to_ f base Hin. apply in_app_or in Hin as [Hin | Hin];
    [ exact (H _ _ _ Hin) | exact (H' _ _ _ Hin) ].
Qed.

Lemma rb_add_facts (l : list fact) (a : acc) :
  reads_blocks_only l -> rb a -> rb (add_facts l a).
Proof. intros Hl Ha. exact (rb_app _ _ Hl Ha). Qed.

(* Facts none of which is an accessor. *)
Lemma rb_none (l : list fact) :
  (∀ f, f ∈ l -> match f with Fact_accessor _ _ _ => False | _ => True end) ->
  reads_blocks_only l.
Proof. intros H to_ f base Hin. exact (False_rect _ (H _ Hin)). Qed.

Lemma rb_map {A : Type} (g : A -> fact) (l : list A) :
  (∀ a, match g a with Fact_accessor _ _ _ => False | _ => True end) ->
  reads_blocks_only (map g l).
Proof.
  intros H. apply rb_none. intros f Hin.
  apply in_map_iff in Hin as (a & <- & _).
  apply H.
Qed.

Lemma note_simple_facts (s : simple) (a : acc) :
  acc_facts (note_simple s a) = acc_facts a.
Proof. destruct s; reflexivity. Qed.

Lemma note_simples_facts (l : list simple) :
  ∀ a : acc, acc_facts (note_simples l a) = acc_facts a.
Proof.
  unfold note_simples. induction l as [ | s l IH]; intros a;
    [ reflexivity | ].
  simpl. rewrite IH. apply note_simple_facts.
Qed.

Lemma fresh_list_facts (n : nat) :
  ∀ a : acc, acc_facts (snd (fresh_list n a)) = acc_facts a.
Proof.
  induction n as [ | n IH]; intros a; [ reflexivity | ]. simpl.
  destruct (fresh_list n _) as [xs a'] eqn:Hl.
  specialize (IH {| acc_next := Pos.succ (acc_next a);
                    acc_facts := acc_facts a; acc_code_deps := acc_code_deps a;
                    acc_apply_deps := acc_apply_deps a;
                    acc_closure_deps := acc_closure_deps a;
                    acc_symbols := acc_symbols a |}).
  rewrite Hl in IH. exact IH.
Qed.

Lemma rb_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  rb a -> rb (cond_any_usage env s a).
Proof.
  intros Ha. unfold cond_any_usage.
  destruct (env_current_code_id env); apply rb_add_facts;
    try (rewrite note_simple_facts; exact Ha);
    apply rb_none; intros f [<- | []]; exact I.
Qed.

Lemma rb_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  rb a -> rb (snd (find_cont env k n a)).
Proof.
  intros Ha. unfold find_cont.
  destruct (lookup k (FlambdaReaperTraversal.env_conts env)); [ exact Ha | ].
  pose proof (fresh_list_facts n a) as Hf.
  destruct (fresh_list n a) as [params a']. simpl in Hf |- *.
  apply rb_add_facts; [ apply rb_map; intros; exact I | ].
  rewrite Hf. exact Ha.
Qed.

Lemma rb_block_deps (base : node) (fields : list simple) (a : acc) :
  rb a -> rb (block_deps base fields a).
Proof.
  intros Ha. unfold block_deps. apply rb_add_facts.
  - apply rb_app; [ apply rb_map; intros [i s]; exact I | ].
    apply rb_none. intros f [<- | [<- | []]]; exact I.
  - rewrite note_simples_facts. exact Ha.
Qed.

Lemma rb_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  rb a -> rb (traverse_prim env x p args a).
Proof.
  intros Ha. unfold traverse_prim.
  assert (Hacc : ∀ f s a, is_block_like_field f = true -> rb a ->
            rb (add_fact (Fact_accessor (Node_variable x) f
                            (node_of_simple s)) (note_simple s a))).
  { intros f s a0 Hf Ha0. apply rb_add_facts.
    - intros to_ f' base [Heq | []]. injection Heq as _ <- _. exact Hf.
    - rewrite note_simple_facts. exact Ha0. }
  assert (Hdefault :
    rb (add_facts
          (Fact_any_source (Node_variable x)
           :: map (fun from => Fact_use (Node_variable x) from)
                (free_names_of_simples args))
          (if prim_has_arbitrary_effects p
           then cond_any_usage env (Var x) a else a))).
  { apply rb_add_facts.
    - apply (rb_app [_]); [ apply rb_none; intros f [<- | []]; exact I | ].
      apply rb_map. intros; exact I.
    - destruct (prim_has_arbitrary_effects p);
        [ apply rb_cond_any_usage, Ha | exact Ha ]. }
  cbv zeta.
  destruct p as [p | p | p | p | p]; [ | exact Hdefault | | | ].
  - destruct p; destruct args as [ | s [ | s' args]]; try exact Hdefault;
      try destruct mut; try destruct variant_only; cbv beta iota;
      first
        [ exact Hdefault
        | apply Hacc; [ reflexivity | exact Ha ]
        | apply rb_add_facts;
          [ apply rb_none; intros f [<- | []]; exact I | ];
          first
            [ apply Hacc; [ reflexivity | exact Ha ]
            | rewrite note_simple_facts; exact Ha ] ].
  - destruct p; [ apply rb_block_deps, Ha | exact Hdefault ].
  - destruct p; try exact Hdefault. apply rb_block_deps, Ha.
  - destruct p; try exact Hdefault. destruct args; exact Ha.
Qed.

Lemma rb_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  rb a -> rb (apply_cont_deps env ac a).
Proof.
  intros Ha. unfold apply_cont_deps.
  pose proof (rb_find_cont env (ac_continuation ac) (length (ac_args ac)) a Ha)
    as Hf.
  destruct (find_cont _ _ _ a) as [params a']. simpl in Hf.
  apply rb_add_facts; [ apply rb_map; intros [p s]; exact I | ].
  rewrite note_simples_facts. exact Hf.
Qed.

Lemma rb_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  rb a -> rb (traverse_switch env scrutinee arms a).
Proof.
  intros Ha. unfold traverse_switch.
  pose proof (rb_cond_any_usage env scrutinee a Ha) as H.
  revert H. generalize (cond_any_usage env scrutinee a).
  induction (Keyed.compact Z.eqb arms) as [ | [d ac] l IH]; intros a' Ha';
    [ exact Ha' | ].
  simpl. apply IH. apply rb_apply_cont_deps, Ha'.
Qed.

Lemma rb_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  rb a -> rb (exn_handler_deps env h a).
Proof.
  intros Ha. unfold exn_handler_deps.
  destruct h as [[ | p params] metadata handler]; [ exact Ha | ].
  destruct (hm_is_exn_handler metadata); [ | exact Ha ].
  apply rb_cond_any_usage, rb_add_facts; [ | exact Ha ].
  apply rb_none. intros f [<- | []]. exact I.
Qed.

(* A unit without closures, projections, calls and code. *)
Lemma rb_traverse :
  (∀ e : expr, closure_free e = true ->
     ∀ (env : traverse_env) (a : acc), rb a -> rb (traverse env e a))
  ∧ (∀ h : continuation_handler, handler_closure_free h = true ->
       ∀ (env : traverse_env) (a : acc), rb a ->
         rb (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hd Hbody].
    change (traverse env (Let x kind d body) a)
      with (traverse env body (traverse_named env x d a)).
    apply (IH Hbody).
    destruct d as [s | p args | | | | | ]; try discriminate Hd.
    + apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
      rewrite note_simple_facts. exact Ha.
    + apply rb_traverse_prim, Ha.
  - intros bindings body _ IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hbs Hbody].
    change (traverse env (Let_static bindings body) a)
      with (traverse env body
              (List.fold_left (fun a b => traverse_binding env b a) bindings
                 (List.fold_left (fun a b => traverse_static_binding env b a)
                    bindings
                    (List.fold_left (fun a b => prepare_binding b a)
                       bindings a)))).
    apply (IH Hbody).
    rewrite forallb_forall in Hbs.
    assert (Hblocks : ∀ b, b ∈ bindings ->
              ∃ s tag fields,
                b = Bind_symbol s (Static_const_block tag fields)).
    { intros b Hb. specialize (Hbs b Hb).
      destruct b as [ | | s [ | | tag fields | | | ]]; try discriminate Hbs.
      eauto. }
    clear Hbs.
    assert (Hprep : ∀ a,
              List.fold_left (fun a b => prepare_binding b a) bindings a = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    assert (Hcode : ∀ a,
              List.fold_left (fun a b => traverse_binding env b a) bindings a
              = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    rewrite Hcode, Hprep. clear Hprep Hcode.
    revert a Ha. induction bindings as [ | b bindings IHb]; intros a Ha;
      [ exact Ha | ].
    destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
    simpl. apply IHb; [ intros b' Hb'; apply Hblocks; right; exact Hb' | ].
    apply rb_block_deps, Ha.
  - intros k h body IHh IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hh Hbody].
    change (traverse env (Let_cont_non_recursive k h body) a)
      with (traverse (add_cont env k (handler_params h)) body
              (exn_handler_deps env h (traverse_handler env h a))).
    apply (IH Hbody).
    apply rb_exn_handler_deps. exact (IHh Hh env a Ha).
  - intros handlers body IHh IH Hcf env a Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hhs Hbody].
    change (traverse env (Let_cont_recursive handlers body) a)
      with (traverse (add_rec_conts env handlers) body
              (List.fold_left
                 (fun a '(_, h) =>
                    traverse_handler (add_rec_conts env handlers) h a)
                 handlers a)).
    apply (IH Hbody).
    rewrite forallb_forall in Hhs.
    revert a Ha. generalize (add_rec_conts env handlers) as env'.
    intros env'. assert (Hsub : ∀ k h, (k, h) ∈ handlers -> (k, h) ∈ handlers)
      by auto. revert Hsub. generalize handlers at 1 3 as hs.
    intros hs Hsub. induction hs as [ | [k h] hs IHhs]; intros a Ha;
      [ exact Ha | ].
    simpl. apply IHhs; [ intros k' h' Hin; apply Hsub; right; exact Hin | ].
    assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
    exact (IHh k h Hin (Hhs _ Hin) env' a Ha).
  - intros ap Hcf. discriminate Hcf.
  - intros ac _ env a Ha. apply rb_apply_cont_deps, Ha.
  - intros scrutinee arms _ env a Ha. apply rb_traverse_switch, Ha.
  - intros _ env a Ha. exact Ha.
  - intros params metadata handler IH Hcf env a Ha. exact (IH Hcf env a Ha).
Qed.

Lemma rb_closure_deps (cds : finite_map code_dep) (ds : list closure_dep) :
  ∀ a : acc,
    rb a -> rb (List.fold_left (fun a d => add_closure_dep cds d a) ds a).
Proof.
  induction ds as [ | d ds IH]; intros a0 Ha0; [ exact Ha0 | ].
  simpl. apply IH. unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) cds) as [cd | ].
  - apply rb_add_facts; [ | exact Ha0 ].
    apply rb_none. intros f Hf.
    destruct Hf as [<- | [<- | Hf]]; [ exact I | exact I | ].
    destruct (cd_unknown_arity_call_witnesses cd); [ destruct Hf | ].
    destruct Hf as [<- | []]. exact I.
  - destruct (fresh a0) as [witness a1] eqn:Hfresh.
    apply rb_add_facts.
    + apply rb_none. intros f Hf.
      repeat (destruct Hf as [<- | Hf]; [ exact I | ]). destruct Hf.
    + unfold fresh in Hfresh. injection Hfresh as _ <-. exact Ha0.
Qed.

Lemma rb_resolve_delayed_deps (symbols : list symbol) (a : acc) :
  rb a -> rb (resolve_delayed_deps symbols a).
Proof.
  intros Ha. unfold resolve_delayed_deps.
  apply rb_add_facts; [ apply rb_map; intros; exact I | ].
  apply rb_closure_deps. apply rb_add_facts; [ | exact Ha ].
  apply rb_none. intros f Hf. apply in_flat_map in Hf as (d & _ & Hf).
  unfold apply_dep_facts in Hf.
  destruct (lookup _ _); [ unfold alias_for_caller in Hf | ];
    destruct (ad_function_containing_apply_expr d);
    repeat (destruct Hf as [<- | Hf]; [ exact I | ]); destruct Hf.
Qed.

Lemma rb_f_accessor (l : list fact) :
  reads_blocks_only l ->
  forallb (fun '(_, f, _) => is_block_like_field f)
    (f_accessor (facts_of_list l))
  = true.
Proof.
  intros Hl. apply forallb_forall. intros [[to_ f] base] Hin.
  cbn [facts_of_list f_accessor] in Hin.
  apply in_flat_map in Hin as (fct & Hfct & Hin).
  destruct fct; simpl in Hin; try contradiction.
  destruct Hin as [[= <- <- <-] | []]. exact (Hl _ _ _ Hfct).
Qed.

(* The graph of a unit without closures, projections, calls and code reads
   only fields of blocks and of boxed numbers. *)
Lemma closure_free_reads_blocks (flag : reaper_preserve_direct_calls)
    (u : FlambdaWellFormedSpec.unit_interface) (e : expr) :
  closure_free e = true ->
  forallb (fun '(_, f, _) => is_block_like_field f)
    (f_accessor (traverse_unit flag u e))
  = true.
Proof.
  intros Hcf. unfold traverse_unit. cbv zeta.
  apply rb_f_accessor. apply rb_resolve_delayed_deps.
  apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
  apply (proj1 rb_traverse e Hcf).
  apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
  apply rb_add_facts; [ apply rb_none; intros f [<- | []]; exact I | ].
  intros to_ f base [].
Qed.

(* The fragment has no closure, projection, call or code. *)
Lemma local_closure_free (unboxed : list (node * unboxed_fields))
    (sol : solution) :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (conts : finite_map (list (variable * bool))),
     FlambdaReaperUnboxingRefinesSpec.local_expr unboxed sol kinds conts e
     = true
     -> closure_free e = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (conts : finite_map (list (variable * bool))) (mask : list bool),
       FlambdaReaperUnboxingRefinesSpec.local_handler unboxed sol kinds conts
         mask h
       = true ->
       handler_closure_free h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hd Hbody].
    rewrite (IH _ _ Hbody), andb_true_r.
    destruct (fields_of unboxed x) as [fields | ].
    + unfold FlambdaReaperUnboxingRefinesSpec.unboxed_binding_ok in Hd.
      destruct kind; try discriminate Hd.
      destruct d; try discriminate Hd; reflexivity.
    + unfold FlambdaReaperUnboxingRefinesSpec.plain_binding_ok in Hd.
      destruct (unboxed_read unboxed d) as [[f fields] | ] eqn:Hread.
      * unfold unboxed_read in Hread. destruct d; try discriminate Hread.
        reflexivity.
      * destruct d; try discriminate Hd; reflexivity.
  - intros bindings body _ IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    rewrite (IH _ _ Hbody), andb_true_r. apply forallb_forall. intros b Hb.
    rewrite forallb_forall in Hbs. specialize (Hbs b Hb).
    destruct b as [ | | s [ | | | | | ]]; try discriminate Hbs. reflexivity.
  - intros k h body IHh IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hh Hbody].
    rewrite (IHh _ _ _ Hh), (IH _ _ Hbody). reflexivity.
  - intros handlers body IHh IH kinds conts Hloc. simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite (IH _ _ Hbody), andb_true_r. apply forallb_forall.
    intros [k h] Hin. rewrite forallb_forall in Hhs.
    exact (IHh k h Hin _ _ _ (Hhs _ Hin)).
  - intros ap kinds conts Hloc. discriminate Hloc.
  - intros ac kinds conts _. reflexivity.
  - intros scrutinee arms kinds conts _. reflexivity.
  - intros kinds conts _. reflexivity.
  - intros params metadata handler IH kinds conts mask Hloc.
    simpl in Hloc |- *.
    apply andb_true_iff in Hloc as [_ Hbody]. exact (IH _ _ Hbody).
Qed.

(* ======================================================================== *)
(* The two rebuilds of a unit of the fragment                               *)
(* ======================================================================== *)

Section Rebuilds.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable first_fresh first_fresh_u : positive.

(* What the graph has of the unit: the facts of FlambdaReaperFlowProofs.v
   ([expr_edges]), and a solution closed under the rules. *)
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable kept_slots : value_slot -> bool.
Variable codes0 : list (code_id * code).
Variable symbols : list symbol.
Variable is_local : field -> bool.
Hypothesis sol_closed : closed is_local (graph_of_facts F) sol.
Hypothesis local_slots : is_local_slots_only is_local.
(* Each unboxed field is read, from a use of the variable, into a node
   that has a use ([get_fields]). *)
Hypothesis fields_read : ∀ (x : variable) (fields : unboxed_fields)
    (f : field) (u : unboxed_field),
  fields_of unboxed x = Some fields -> (f, u) ∈ fields ->
  ∃ w base : node,
    g_accessor (graph_of_facts F) w f base
    ∧ s_usages sol (Node_variable x) base = true
    ∧ s_has_usage sol w = true.
(* A read that the rebuild keeps, into a variable that is not unboxed,
   of an unboxed variable, finds its field among the unboxed ones, a
   variable ([check_unboxed_reads]). *)
Hypothesis reads_found : ∀ (x y : variable) (kind : flambda_kind)
    (f : field) (fields : unboxed_fields),
  fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
  g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
  is_dead_var sol x kind = false -> is_var_used sol x kind = true ->
  kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
  ∃ v : variable, Keyed.lookup field_eqb f fields = Some (Not_unboxed v).

Abbreviation edges ce e :=
  (FlambdaReaperFlowProofs.expr_edges (graph_of_facts F) noted deps cds
     kept_slots codes0 symbols True False None ce e).

(* A simple that [rewrite_simple] keeps. *)
Definition kept_simple (en : rebuild_env) (s : simple) : bool :=
  match s with
  | Var x => is_var_used sol x (kind_of_variable en x)
  | Symbol sym => is_symbol_used sol sym
  | Const _ => true
  end.

(* A field unboxed is used, and what the allocation holds there is kept:
   the solution has it of the field read. *)
Lemma field_kept (x : variable) (fields : unboxed_fields) (f : field)
    (v : unboxed_field) (a : simple) (en : rebuild_env) :
  fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
  g_constructor (graph_of_facts F) (Node_variable x) f (node_of_simple a) ->
  is_block_like_field f = true ->
  s_field_used sol (Node_variable x) f = true ∧ kept_simple en a = true.
Proof.
  intros Hx Hin Hcons Hblock.
  destruct (fields_read x fields f v Hx Hin) as (w & base & Hacc & Hus & Hw).
  assert (Hlocal : is_local f = false).
  { destruct (is_local f) eqn:Hl; [ | reflexivity ].
    destruct (local_slots f Hl) as [(w' & ->) | (fs & ->)];
      discriminate Hblock. }
  split.
  - exact (field_used_usage _ _ _ sol_closed _ _ _ _ _ Hcons Hus Hacc Hw).
  - assert (Hhu : s_has_usage sol (node_of_simple a) = true).
    { destruct (s_any_usage sol (Node_variable x)) eqn:Hau.
      - apply (has_usage_any _ _ _ sol_closed).
        exact (any_usage_constructor _ _ _ sol_closed _ _ _ Hau Hcons Hlocal).
      - apply (has_usage_alias _ _ _ sol_closed w); [ exact Hw | ].
        exact (alias_constructor_usage _ _ _ sol_closed _ _ _ _ _ Hcons Hus
                 Hau Hacc). }
    destruct a as [y | sym | c]; simpl in Hhu |- *; [ | | reflexivity ].
    + unfold is_var_used. destruct (kind_of_variable en y); try reflexivity;
        exact Hhu.
    + exact Hhu.
Qed.

Abbreviation sol_u := (solution_without_unboxed sol unboxed).

Definition plain_rebuild (en : rebuild_env) (e : expr) : expr :=
  rebuild_expr F sol preserve_direct_calls unit_code_ids first_fresh en e.

Definition unbox_rebuild (en : unbox_env) (e : expr) : expr :=
  rebuild_expr_u false false F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes unboxed first_fresh_u [] en e.

Definition is_decided (d : decision) : bool :=
  match d with
  | Decided _ => true
  | Unbox _ | Keep_unused _ => false
  end.

(* The handlers in scope as the fragment has them ([C]): their
   parameters, with what the plain rebuild keeps of each. The plain
   rebuild has their masks ([pen_of]), the rebuild with unboxing their
   decisions ([conts_match]), and [ub] the shapes of the parameters that
   the plain rebuild keeps ([K_of]). *)
Definition pen_of (en : unbox_env) (C : finite_map (list (variable * bool)))
    : rebuild_env :=
  {| env_conts := map (fun '(k, ps) => (k, map snd ps)) C;
     env_kinds := env_kinds (ue_env en);
     env_projections := [] |}.

Definition shape_of_param (pk : variable * bool)
    : list (option unboxed_fields) :=
  let '(p, keep) := pk in if keep then [fields_of unboxed p] else [].

Definition K_of (C : finite_map (list (variable * bool)))
    : finite_map (list (option unboxed_fields)) :=
  map (fun '(k, ps) => (k, flat_map shape_of_param ps)) C.

Definition decision_match (pk : variable * bool) (d : decision) : Prop :=
  let '(p, keep) := pk in
  match fields_of unboxed p with
  | Some fields => d = Unbox fields ∧ (keep = true ∨ fields = [])
  | None =>
    (d = Decided Delete ∧ keep = false)
    ∨ ((∃ kind : flambda_kind, d = Decided (Keep kind)) ∧ keep = true)
  end.

Definition conts_match (C : finite_map (list (variable * bool)))
    (D : finite_map (list decision)) : Prop :=
  Forall2
    (fun (kp : continuation * list (variable * bool))
         (kd : continuation * list decision) =>
       fst kd = fst kp ∧ Forall2 decision_match (snd kp) (snd kd))
    C D.

(* No projection is pending. *)
Record uinv (en : unbox_env) : Prop := {
  uinv_drop : ue_drop_projections en = false;
  uinv_projections : env_projections (ue_env en) = []
}.


Lemma plain_not_bound (s : simple) :
  plain_simple unboxed s = true -> unboxed_of unboxed s = None.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs; [ | | reflexivity ].
  - apply plain_variable_iff in Hs as [Hs _]. unfold Keyed.is_bound in Hs.
    unfold unboxed_of_node. destruct (Keyed.lookup _ _ _); congruence.
  - apply negb_true_iff in Hs. unfold Keyed.is_bound in Hs.
    unfold unboxed_of_node. destruct (Keyed.lookup _ _ _); congruence.
Qed.

Lemma rewrite_simple_u (en : rebuild_env) (s : simple) :
  plain_simple unboxed s = true ->
  rewrite_simple sol_u en s = rewrite_simple sol en s.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs; [ | | reflexivity ].
  - apply plain_variable_iff in Hs as [Hs _].
    unfold is_var_used. destruct (kind_of_variable en x); try reflexivity;
      cbn [solution_without_unboxed s_has_usage is_unboxed_name];
      rewrite Hs, andb_true_r; reflexivity.
  - apply negb_true_iff in Hs. unfold is_symbol_used.
    cbn [solution_without_unboxed s_has_usage is_unboxed_name].
    rewrite Hs, andb_true_r. reflexivity.
Qed.

(* What the plain rebuild makes of a plain simple is plain. *)
Lemma rewrite_simple_plain (en : rebuild_env) (s : simple) :
  plain_simple unboxed s = true ->
  plain_simple unboxed (rewrite_simple sol en s) = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hs; [ | | reflexivity ].
  - destruct (is_var_used sol x _); [ exact Hs | reflexivity ].
  - destruct (is_symbol_used sol sym); [ exact Hs | reflexivity ].
Qed.

Lemma rewrite_simples_plain (en : rebuild_env) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  forallb (plain_simple unboxed) (map (rewrite_simple sol en) args) = true.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite !andb_true_iff. intros [Hs Hargs].
  split; [ apply rewrite_simple_plain, Hs | apply IH, Hargs ].
Qed.

Lemma rebuild_fields_u (en : rebuild_env) (base : node) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  ∀ i : nat,
    rebuild_fields sol_u en base i args = rebuild_fields sol en base i args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite andb_true_iff. intros [Hs Hargs] i.
  rewrite (rewrite_simple_u en s Hs), (IH Hargs). reflexivity.
Qed.

Lemma rebuild_fields_plain (en : rebuild_env) (base : node)
    (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  ∀ i : nat,
    forallb (plain_simple unboxed) (rebuild_fields sol en base i args) = true.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite andb_true_iff. intros [Hs Hargs] i. rewrite andb_true_iff.
  split; [ | apply IH, Hargs ].
  destruct (field_used sol base _); [ apply rewrite_simple_plain, Hs | ].
  reflexivity.
Qed.

Lemma rebuild_fields_nth (en : rebuild_env) (base : node)
    (args : list simple) :
  ∀ (k i : nat) (s : simple),
    nth_error args i = Some s ->
    nth_error (rebuild_fields sol en base k args) i
    = Some (if field_used sol base (Field_block (k + i) Value)
            then rewrite_simple sol en s
            else Const (Poison Value)).
Proof.
  induction args as [ | a args IH]; intros k i s Hnth;
    [ destruct i; discriminate Hnth | ].
  destruct i as [ | i]; simpl in Hnth |- *.
  - injection Hnth as ->. rewrite Nat.add_0_r. reflexivity.
  - rewrite (IH (S k) i s Hnth). rewrite Nat.add_succ_r. reflexivity.
Qed.

(* The plain rebuild of a [Let] that is not of an unboxed variable, nor a
   read of one. *)
Lemma named_plain_u (en : rebuild_env) (x : variable) (d : named) :
  named_plain unboxed d = true ->
  rebuild_named sol_u en x d = rebuild_named sol en x d.
Proof.
  intros Hd. destruct d as [s | p args | | | | | ]; simpl in Hd;
    try discriminate Hd.
  - simpl. rewrite (rewrite_simple_u en s Hd). reflexivity.
  - assert (Hmap : map (rewrite_simple sol_u en) args
                   = map (rewrite_simple sol en) args).
    { apply map_ext_in. intros s Hs. apply rewrite_simple_u.
      rewrite forallb_forall in Hd. exact (Hd s Hs). }
    destruct p as [p | p | p | p | p]; destruct p; simpl; rewrite ?Hmap;
      rewrite ?(rebuild_fields_u en (Node_variable x) args Hd 0);
      reflexivity.
Qed.

Lemma named_plain_rebuilt (en : rebuild_env) (x : variable) (d : named) :
  named_plain unboxed d = true ->
  named_plain unboxed (rebuild_named sol en x d) = true.
Proof.
  intros Hd. destruct d as [s | p args | | | | | ]; simpl in Hd;
    try discriminate Hd.
  - simpl. apply rewrite_simple_plain, Hd.
  - destruct p as [p | p | p | p | p]; simpl;
      try (apply rewrite_simples_plain, Hd).
    + destruct p; simpl; try (apply rewrite_simples_plain, Hd).
      destruct (field_used sol _ _); simpl;
        [ apply rewrite_simples_plain, Hd | reflexivity ].
    + destruct p; simpl; try (apply rewrite_simples_plain, Hd).
      apply rebuild_fields_plain, Hd.
    + destruct p; simpl; try (apply rewrite_simples_plain, Hd).
      apply rebuild_fields_plain, Hd.
Qed.

Lemma lets_free_continuations (l : list (variable * flambda_kind * simple))
    : ∀ body : expr,
      free_continuations_of_expr (lets l body) = free_continuations_of_expr body
      ∧ binds_statics (lets l body) = binds_statics body.
Proof.
  induction l as [ | [[v kind] s] l IH]; intros body; [ split; reflexivity | ].
  change (lets ((v, kind, s) :: l) body)
    with (lets l (Let v kind (Simple s) body)).
  destruct (IH (Let v kind (Simple s) body)) as [H1 H2].
  rewrite H1, H2. split; reflexivity.
Qed.

Lemma kinds_env_kept (en : rebuild_env) (s : simple) :
  kept_simple (kinds_env (env_kinds en)) s = kept_simple en s.
Proof. destruct s; reflexivity. Qed.

Lemma kinds_env_kind (en : rebuild_env) (s : simple) :
  simple_kind (kinds_env (env_kinds en)) s = simple_kind en s.
Proof. destruct s; reflexivity. Qed.

Lemma kept_rewrite (en : rebuild_env) (s : simple) :
  kept_simple en s = true -> rewrite_simple sol en s = s.
Proof.
  destruct s as [x | sym | c]; simpl; intros H; [ | | reflexivity ];
    rewrite H; reflexivity.
Qed.

(* The field of a boxed number that is unboxed. *)
Lemma bind_boxed_number_lets (x : variable) (bn : boxable_number)
    (c : simple) (fields : unboxed_fields) (d : named) :
  plain_simple unboxed c = true ->
  (fields <> [] -> d = Prim (Unary (Box_number bn)) [c]) ->
  (∀ (f : field) (u : unboxed_field),
     (f, u) ∈ fields ->
     ∃ v : variable, f = Field_boxed_number bn ∧ u = Not_unboxed v) ->
  ∀ body : expr,
    ∃ ls : list (list (variable * flambda_kind * simple)),
      bind_boxed_number_field bn c fields body = Some (lets (concat ls) body)
      ∧ Forall2 (alloc_entry unboxed x d) fields ls.
Proof.
  intros Hc Hd. unfold bind_boxed_number_field.
  induction fields as [ | [f u] fields IH]; intros Hfields body.
  - exists []. split; [ reflexivity | constructor ].
  - destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
    assert (Hd' : d = Prim (Unary (Box_number bn)) [c])
      by (apply Hd; discriminate).
    destruct (IH (fun _ => Hd') (fun f' u' Hin => Hfields f' u' (or_intror Hin))
                (Let v (unboxed_kind bn) (Simple c) body))
      as (ls & Hl & Hentries).
    exists ([(v, unboxed_kind bn, c)] :: ls). split.
    + simpl List.fold_left. rewrite Nat.eqb_refl. exact Hl.
    + constructor; [ | exact Hentries ]. subst d. cbn [alloc_entry].
      exists (unboxed_kind bn), c.
      split; [ reflexivity | split; [ | exact Hc ] ].
      split; [ reflexivity | split; reflexivity ].
Qed.

Lemma with_kinds_combine (params : list variable)
    (kinds : list flambda_kind) :
  length params = length kinds ->
  with_kinds params kinds = combine params kinds.
Proof.
  revert kinds. induction params as [ | p params IH]; intros [ | k kinds] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma params_to_keep_length (is_exn : bool)
    (bound : list (variable * flambda_kind)) :
  length (params_to_keep sol is_exn bound) = length bound.
Proof. unfold params_to_keep. apply length_map. Qed.

(* The arguments of an [Apply_cont]. *)
Lemma get_args_decided (en : rebuild_env) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  ∀ ds : list decision,
    forallb is_decided ds = true ->
    get_args sol unboxed en ds args
    = Some (filter_kept (map is_kept_decision ds)
              (map (rewrite_simple sol en) args)).
Proof.
  induction args as [ | a args IH]; intros Hargs ds Hds; [ reflexivity | ].
  simpl in Hargs. apply andb_true_iff in Hargs as [Ha Hargs].
  destruct ds as [ | d ds].
  - simpl. rewrite (IH Hargs [] eq_refl), (rewrite_simple_u en a Ha).
    reflexivity.
  - simpl in Hds. apply andb_true_iff in Hds as [Hd Hds].
    simpl. rewrite (IH Hargs ds Hds).
    destruct d as [[kind | ] | fields | kind']; simpl in Hd;
      try discriminate Hd;
      simpl; [ rewrite (rewrite_simple_u en a Ha) | ]; reflexivity.
Qed.

Lemma map_fst_combine {A B : Type} (l : list A) (l' : list B) :
  length l = length l' -> map fst (combine l l') = l.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l'] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma map_snd_combine {A B : Type} (l : list A) (l' : list B) :
  length l = length l' -> map snd (combine l l') = l'.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l'] Hlen;
    simpl in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma filter_kept_forallb {A : Type} (p : A -> bool) (mask : list bool)
    (l : list A) :
  forallb p l = true -> forallb p (filter_kept mask l) = true.
Proof.
  revert mask. induction l as [ | a l IH]; intros mask Hl; [ reflexivity | ].
  simpl in Hl. apply andb_true_iff in Hl as [Ha Hl].
  destruct mask as [ | [ | ] mask]; simpl; rewrite ?Ha, ?IH by exact Hl;
    try reflexivity; apply IH, Hl.
Qed.

Lemma lookup_map_values {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k', a) => (k', f a)) l) = option_map f (lookup k l).
Proof.
  induction l as [ | [k' a] l IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* The [Let] of an unboxed variable, as the fragment has it. *)
Lemma unboxed_binding_cases (kinds : finite_map flambda_kind) (x : variable)
    (kind : flambda_kind) (d : named) (fields : unboxed_fields) :
  unboxed_binding_ok unboxed sol kinds x kind d fields = true ->
  kind = Value
  ∧ ((∃ (tag : Z) (args : list simple),
        d = Prim (Variadic (Make_block (Block_kind_values tag))) args
        ∧ ∀ (f : field) (u : unboxed_field),
            (f, u) ∈ fields ->
            ∃ (i : nat) (a : simple),
              f = Field_block i Value ∧ nth_error args i = Some a
              ∧ simple_kind (kinds_env kinds) a = Value
              ∧ (((∃ v : variable, u = Not_unboxed v)
                  ∧ plain_simple unboxed a = true)
                 ∨ ∃ (nested : unboxed_fields) (b : variable)
                     (fields_b : unboxed_fields),
                     u = Unboxed nested ∧ a = Var b ∧ x <> b
                     ∧ fields_of unboxed b = Some fields_b
                     ∧ fields_alike nested fields_b = true))
     ∨ (∃ (bn : boxable_number) (c : simple),
          d = Prim (Unary (Box_number bn)) [c]
          ∧ plain_simple unboxed c = true
          ∧ ∀ (f : field) (u : unboxed_field),
              (f, u) ∈ fields ->
              ∃ v : variable, f = Field_boxed_number bn ∧ u = Not_unboxed v)
     ∨ (∃ (y : variable) (fields_y : unboxed_fields),
          d = Simple (Var y) ∧ x <> y ∧ fields_of unboxed y = Some fields_y
          ∧ fields_alike fields fields_y = true
          ∧ is_var_used sol y (kind_of_variable (kinds_env kinds) y) = true)
     ∨ (∃ (bk : block_access_kind) (mut : mutability) (i : Z) (y : variable)
          (fields_y nested : unboxed_fields),
          d = Prim (Unary (Block_load bk mut i)) [Var y] ∧ x <> y
          ∧ fields_of unboxed y = Some fields_y
          ∧ Keyed.lookup field_eqb (Field_block (Z.to_nat i) Value) fields_y
            = Some (Unboxed nested)
          ∧ fields_alike fields nested = true)).
Proof.
  unfold unboxed_binding_ok. intros H.
  destruct kind; try discriminate H. split; [ reflexivity | ].
  destruct d as [s | p args | | | | | ]; try discriminate H.
  { destruct s as [y | | ]; try discriminate H.
    right. right. left. apply andb_true_iff in H as [Hne H].
    unfold unboxed_arg_ok in H.
    destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
      [ | discriminate H ].
    apply andb_true_iff in H as [Hflat Hused].
    exists y, fields_y. split; [ reflexivity | ].
    split; [ intros ->; rewrite Pos.eqb_refl in Hne; discriminate Hne | ].
    split; [ exact Hy | split; [ exact Hflat | exact Hused ] ]. }
  destruct p as [p | p | p | p | p]; try discriminate H.
  - destruct p; try discriminate H.
    + destruct args as [ | [y | | ] [ | ]]; try discriminate H.
      right. right. right. apply andb_true_iff in H as [Hne H].
      destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
        [ | discriminate H ].
      destruct (Keyed.lookup field_eqb _ fields_y) as [[ | nested] | ] eqn:Hn;
        try discriminate H.
      eexists _, _, _, y, fields_y, nested. split; [ reflexivity | ].
      split; [ intros ->; rewrite Pos.eqb_refl in Hne; discriminate Hne | ].
      split; [ exact Hy | split; [ exact Hn | exact H ] ].
    + rename kind into bn.
      destruct args as [ | c [ | ]]; try discriminate H.
      right. left. apply andb_true_iff in H as [Hc Hfields].
      exists bn, c. split; [ reflexivity | split; [ exact Hc | ] ].
      intros f u Hin. rewrite forallb_forall in Hfields.
      specialize (Hfields (f, u) Hin). cbn beta iota in Hfields.
      destruct u as [v | ]; [ | discriminate Hfields ].
      apply field_eqb_eq in Hfields. subst f.
      exists v. split; reflexivity.
  - destruct p as [[tag] | ]; try discriminate H.
    left. exists tag, args. split; [ reflexivity | ].
    intros f u Hin. rewrite forallb_forall in H.
    specialize (H (f, u) Hin). cbn beta iota in H.
    destruct f as [i k | | | | | | | | ]; try discriminate H.
    destruct k; try discriminate H.
    destruct u as [v | nested].
    + destruct (nth_error args i) as [a | ] eqn:Hnth; [ | discriminate H ].
      apply andb_true_iff in H as [Ha Hk]. apply kind_eqb_eq in Hk.
      exists i, a. split; [ reflexivity | split; [ exact Hnth | ] ].
      split; [ exact Hk | ]. left. split; [ exists v; reflexivity | exact Ha ].
    + destruct (nth_error args i) as [[b | | ] | ] eqn:Hnth;
        try discriminate H.
      apply andb_true_iff in H as [H Hok].
      apply andb_true_iff in H as [Hne Hk]. apply kind_eqb_eq in Hk.
      unfold unboxed_arg_ok in Hok.
      destruct (fields_of unboxed b) as [fields_b | ] eqn:Hb;
        [ | discriminate Hok ].
      apply andb_true_iff in Hok as [Halike _].
      exists i, (Var b). split; [ reflexivity | split; [ exact Hnth | ] ].
      split; [ exact Hk | ]. right. exists nested, b, fields_b.
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ intros ->; rewrite Pos.eqb_refl in Hne; discriminate Hne | ].
      split; [ exact Hb | exact Halike ].
Qed.

(* The [Let] of a variable that is not unboxed, as the fragment has
   it. *)
Lemma plain_binding_cases (x : variable) (kind : flambda_kind) (d : named) :
  plain_binding_ok unboxed x kind d = true ->
  named_plain unboxed d = true
  ∨ ∃ (p : unary_primitive) (y : variable) (f : field)
      (fields : unboxed_fields),
      d = Prim (Unary p) [Var y]
      ∧ read_of (Prim (Unary p) [Var y]) = Some (f, Var y)
      ∧ (∀ s : simple, read_of (Prim (Unary p) [s]) = Some (f, s))
      ∧ (∀ en : rebuild_env,
           rebuild_named sol en x d
           = Prim (Unary p) [rewrite_simple sol en (Var y)])
      ∧ fields_of unboxed y = Some fields
      ∧ (kind = Value ∨ ∃ nk : naked_number_kind, kind = Naked_number nk).
Proof.
  unfold plain_binding_ok. destruct (unboxed_read unboxed d) as [[f fields] | ]
    eqn:Hread.
  - intros H. right. unfold unboxed_read in Hread.
    destruct d as [ | prim args | | | | | ]; try discriminate Hread.
    destruct prim as [p | | | | ]; try discriminate Hread.
    destruct args as [ | [y | | ] [ | ]]; try discriminate Hread;
      destruct p; try discriminate Hread;
      simpl in Hread;
      destruct (fields_of unboxed y) as [fs | ] eqn:Hy; try discriminate Hread;
      injection Hread as <- <-;
      eexists _, y, _, fs;
      (split; [ reflexivity | split; [ reflexivity | split;
        [ intros s; reflexivity | split; [ intros en; reflexivity | ] ] ] ]);
      (split; [ exact Hy | ]);
      (destruct kind as [ | nk | | ]; try discriminate H;
       [ left; reflexivity | right; exists nk; reflexivity ]).
  - intros H. left. destruct d; exact H.
Qed.

Lemma pen_projections (en : unbox_env)
    (C : finite_map (list (variable * bool))) :
  env_projections (pen_of en C) = [].
Proof. reflexivity. Qed.

Lemma rewrite_simple_kinds (en1 en2 : rebuild_env) (s : simple) :
  env_kinds en1 = env_kinds en2 ->
  rewrite_simple sol en1 s = rewrite_simple sol en2 s.
Proof.
  intros H. destruct s; simpl; [ | reflexivity | reflexivity ].
  unfold kind_of_variable. rewrite H. reflexivity.
Qed.

Lemma rebuild_fields_kinds (en1 en2 : rebuild_env) (base : node)
    (args : list simple) :
  env_kinds en1 = env_kinds en2 ->
  ∀ i : nat,
    rebuild_fields sol en1 base i args = rebuild_fields sol en2 base i args.
Proof.
  intros H. induction args as [ | s args IH]; intros i; simpl;
    [ reflexivity | ].
  rewrite (rewrite_simple_kinds en1 en2 s H), IH. reflexivity.
Qed.

Lemma rebuild_named_kinds (en1 en2 : rebuild_env) (x : variable)
    (d : named) :
  env_kinds en1 = env_kinds en2 ->
  rebuild_named sol en1 x d = rebuild_named sol en2 x d.
Proof.
  intros H.
  assert (Hm : ∀ l, map (rewrite_simple sol en1) l
                    = map (rewrite_simple sol en2) l)
    by (intros l; apply map_ext; intros s; apply rewrite_simple_kinds, H).
  destruct d as [s | p args | | | | | ]; simpl;
    rewrite ?(rewrite_simple_kinds en1 en2 _ H); try reflexivity.
  destruct p as [p | p | p | p | p]; destruct p; simpl; rewrite ?Hm;
    rewrite ?(rebuild_fields_kinds en1 en2 _ args H 0); reflexivity.
Qed.

Lemma uinv_bind (en : unbox_env) (bound : list (variable * flambda_kind)) :
  uinv (ubind_variables en bound).
Proof. split; reflexivity. Qed.

Lemma uinv_continuations (en : unbox_env)
    (bound : list (continuation * list decision)) :
  uinv (ubind_continuations en bound).
Proof. split; reflexivity. Qed.

Lemma conts_match_lookup (C : finite_map (list (variable * bool)))
    (D : finite_map (list decision)) :
  conts_match C D ->
  ∀ k : continuation,
    (lookup k C = None ∧ lookup k D = None)
    ∨ ∃ (ps : list (variable * bool)) (ds : list decision),
        lookup k C = Some ps ∧ lookup k D = Some ds
        ∧ Forall2 decision_match ps ds.
Proof.
  induction 1 as [ | [k' ps] [k'' ds] C D [Hk Hds] _ IH]; intros k;
    [ left; split; reflexivity | ].
  simpl in Hk, Hds. subst k''. rewrite !lookup_cons.
  destruct (Pos.eqb k k').
  - right. exists ps, ds. split; [ reflexivity | split; [ reflexivity | ] ].
    exact Hds.
  - exact (IH k).
Qed.

(* The variables of the fields of an argument unboxed alike. *)
Lemma unboxed_args_pairs (fields : unboxed_fields) (y : variable)
    (fields_y : unboxed_fields) :
  fields_of unboxed y = Some fields_y ->
  fields_alike fields fields_y = true ->
  ∃ pairs : list (variable * variable * flambda_kind),
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs
    ∧ unboxed_args unboxed fields (Var y)
      = Some (map (fun '(_, x, kind) => (Var x, kind)) pairs).
Proof.
  intros Hy Hf. unfold fields_alike in Hf.
  destruct (unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y))
    as [pairs | ] eqn:Hp; [ | discriminate Hf ].
  exists pairs. split; [ reflexivity | ].
  unfold unboxed_args. cbn [unboxed_of].
  change (unboxed_of_node unboxed (Node_variable y))
    with (fields_of unboxed y).
  rewrite Hy, Hp. reflexivity.
Qed.

(* The arguments of an [Apply_cont] to a handler of the unit. *)
Lemma get_args_rel (en : rebuild_env) (args : list simple) :
  ∀ (ps : list (variable * bool)) (ds : list decision),
    Forall2 decision_match ps ds ->
    args_ok unboxed sol (env_kinds en) ps args = true ->
    ∃ args_u : list simple,
      get_args sol unboxed en ds args = Some args_u
      ∧ args_rel unboxed (flat_map shape_of_param ps)
          (filter_kept (map snd ps) (map (rewrite_simple sol en) args))
          args_u.
Proof.
  induction args as [ | a args IH]; intros ps ds Hds Hok.
  - exists []. split; [ reflexivity | ]. cbn [map].
    replace (filter_kept (map snd ps) []) with (@nil simple)
      by (destruct (map snd ps); reflexivity).
    destruct (flat_map shape_of_param ps) as [ | o sh ];
      [ apply args_rel_beyond; reflexivity | apply args_rel_missing ].
  - destruct Hds as [ | [p keep] d ps ds Hd Hds ].
    + exists (filter_kept [] (map (rewrite_simple sol en) (a :: args))).
      split; [ exact (get_args_decided en (a :: args) Hok [] eq_refl) | ].
      apply args_rel_beyond. apply filter_kept_forallb, rewrite_simples_plain.
      exact Hok.
    + simpl in Hok. apply andb_true_iff in Hok as [Ha Hok].
      destruct (IH ps ds Hds Hok) as (rest & Hrest & Hrel).
      cbn [get_args]. rewrite Hrest.
      unfold decision_match in Hd.
      cbn [flat_map shape_of_param map filter_kept].
      destruct (fields_of unboxed p) as [fields | ] eqn:Hp.
      * destruct Hd as [-> Hkeep].
        unfold unboxed_arg_ok in Ha.
        destruct a as [y | sym | c]; try discriminate Ha.
        destruct (fields_of unboxed y) as [fields_y | ] eqn:Hy;
          [ | discriminate Ha ].
        apply andb_true_iff in Ha as [Hflat Hused].
        destruct (unboxed_args_pairs fields y fields_y Hy Hflat)
          as (pairs & Hpairs & Hl).
        rewrite Hl. cbn [option_map].
        exists (map (fun '(_, x', _) => Var x') pairs ++ rest).
        split; [ rewrite map_map; do 2 f_equal; apply map_ext;
                 intros [[a b] k]; reflexivity | ].
        destruct keep.
        -- cbn [app]. simpl in Hused. unfold kinds_env in Hused.
           cbn [rewrite_simple]. unfold kind_of_variable in Hused |- *.
           cbn [env_kinds] in Hused. rewrite Hused.
           exact (args_rel_unboxed unboxed _ fields fields_y y pairs _ _ Hy
                    Hpairs Hrel).
        -- destruct Hkeep as [Hk | ->]; [ discriminate Hk | ].
           cbn in Hpairs. injection Hpairs as <-. exact Hrel.
      * destruct Hd as [[-> ->] | [(kind & ->) ->]].
        -- exists rest. split; [ reflexivity | exact Hrel ].
        -- simpl in Ha. exists (rewrite_simple sol_u en a :: rest).
           split; [ reflexivity | ].
           rewrite (rewrite_simple_u en a Ha).
           apply args_rel_plain; [ apply rewrite_simple_plain, Ha | ].
           exact Hrel.
Qed.

Lemma rewrite_apply_cont_rel (en : unbox_env)
    (C : finite_map (list (variable * bool))) (ac : apply_cont) :
  conts_match C (ue_decisions en) ->
  apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true ->
  (rewrite_apply_cont sol (pen_of en C) ac = None
   ∧ rewrite_apply_cont_u sol unboxed en ac = None)
  ∨ ∃ ac_p ac_u : apply_cont,
      rewrite_apply_cont sol (pen_of en C) ac = Some ac_p
      ∧ rewrite_apply_cont_u sol unboxed en ac = Some ac_u
      ∧ ac_rel unboxed (K_of C) ac_p ac_u.
Proof.
  intros HC Hok. unfold rewrite_apply_cont, rewrite_apply_cont_u.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ left; split; reflexivity | right ].
  unfold apply_cont_ok in Hok.
  assert (Hmap : map (rewrite_simple sol (pen_of en C)) (ac_args ac)
                 = map (rewrite_simple sol (ue_env en)) (ac_args ac))
    by (apply map_ext; intros s; apply rewrite_simple_kinds; reflexivity).
  rewrite Hmap. cbn [pen_of env_conts]. rewrite lookup_map_values.
  destruct (conts_match_lookup C _ HC (ac_continuation ac))
    as [[HCk HDk] | (ps & ds & HCk & HDk & Hds)];
    rewrite HCk in Hok |- *; rewrite HDk; cbn [option_map].
  - assert (Hu : map (rewrite_simple sol_u (ue_env en)) (ac_args ac)
                 = map (rewrite_simple sol (ue_env en)) (ac_args ac)).
    { apply map_ext_in. intros s Hs. apply rewrite_simple_u.
      rewrite forallb_forall in Hok. exact (Hok s Hs). }
    rewrite Hu. eexists _, _. split; [ reflexivity | split; [ reflexivity | ] ].
    unfold ac_rel, shapes_at, K_of.
    cbn [ac_continuation ac_args ac_trap_action].
    rewrite lookup_map_values, HCk. cbn [option_map].
    split; [ reflexivity | split; [ reflexivity | ] ].
    apply args_rel_beyond, rewrite_simples_plain, Hok.
  - destruct (get_args_rel (ue_env en) (ac_args ac) ps ds Hds Hok)
      as (args_u & Hget & Hrel).
    rewrite Hget. eexists _, _.
    split; [ reflexivity | split; [ reflexivity | ] ].
    unfold ac_rel, shapes_at, K_of.
    cbn [ac_continuation ac_args ac_trap_action].
    rewrite lookup_map_values, HCk. cbn [option_map].
    split; [ reflexivity | split; [ reflexivity | exact Hrel ] ].
Qed.

Lemma with_kinds_length (params : list variable) :
  ∀ kinds : list flambda_kind, length (with_kinds params kinds) = length params.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds]; simpl;
    rewrite ?IH; reflexivity.
Qed.

Lemma hm_with_params_kinds (metadata : handler_metadata)
    (kinds : list flambda_kind) :
  hm_params_kinds (with_params_kinds metadata kinds) = kinds.
Proof.
  unfold with_params_kinds. destruct (hm_is_exn_handler metadata); reflexivity.
Qed.

(* The parameters of a handler as the two rebuilds make them. *)
Lemma params_rel_handler (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    forallb (fun '((p, kind), keep) => param_ok unboxed p kind keep)
      (combine bound mask)
    = true ->
    params_rel unboxed (flat_map shape_of_param (combine (map fst bound) mask))
      (filter_kept mask bound)
      (params_of_decisions bound
         (map
            (fun (p : (variable * flambda_kind) * bool) =>
               let '((x, kind), keep) := p in
               match fields_of unboxed x with
               | Some fields => Unbox fields
               | None => Decided (if keep then Keep kind else Delete)
               end)
            (combine bound mask)))
    ∧ Forall2 decision_match (combine (map fst bound) mask)
        (map
           (fun (p : (variable * flambda_kind) * bool) =>
              let '((x, kind), keep) := p in
              match fields_of unboxed x with
              | Some fields => Unbox fields
              | None => Decided (if keep then Keep kind else Delete)
              end)
           (combine bound mask)).
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen Hok;
    simpl in Hlen; try discriminate Hlen.
  - split; constructor.
  - simpl in Hok. apply andb_true_iff in Hok as [Hx Hok].
    destruct (IH mask ltac:(lia) Hok) as [Hrel Hmatch].
    unfold param_ok in Hx. apply andb_true_iff in Hx as [Hfresh Hx].
    apply negb_true_iff in Hfresh.
    unfold FlambdaReaperUnboxingRefinesSpec.param_ok in Hx.
    cbn [map fst combine flat_map shape_of_param].
    destruct (fields_of unboxed x) as [fields | ] eqn:Hf.
    + apply andb_true_iff in Hx as [Hkind Hkeep].
      apply kind_eqb_eq in Hkind. subst kind.
      split.
      * destruct keep.
        -- cbn [app filter_kept params_of_decisions decision_params].
           apply params_rel_unboxed; assumption.
        -- destruct fields; [ | discriminate Hkeep ].
           cbn [app filter_kept params_of_decisions decision_params
                fields_leaves unboxed_field_leaves].
           exact Hrel.
      * constructor; [ | exact Hmatch ]. unfold decision_match. rewrite Hf.
        split; [ reflexivity | ].
        destruct keep; [ left; reflexivity | right ].
        destruct fields; [ reflexivity | discriminate Hkeep ].
    + split.
      * destruct keep;
          cbn [app filter_kept params_of_decisions decision_params];
          [ | exact Hrel ].
        apply params_rel_plain; [ | exact Hrel ].
        apply plain_variable_iff. split; [ | exact Hfresh ].
        unfold Keyed.is_bound. unfold fields_of in Hf. rewrite Hf.
        reflexivity.
      * constructor; [ | exact Hmatch ]. unfold decision_match. rewrite Hf.
        destruct keep; [ right; split; [ eexists; reflexivity | reflexivity ]
                       | left; split; reflexivity ].
Qed.

Lemma handler_match (kinds : finite_map flambda_kind)
    (C : finite_map (list (variable * bool))) (is_exn : bool)
    (h : continuation_handler) :
  local_handler unboxed sol kinds C (handler_mask sol is_exn h) h = true ->
  Forall2 decision_match
    (combine (handler_variables h) (handler_mask sol is_exn h))
    (handler_decisions false sol unboxed is_exn h)
  ∧ length (handler_mask sol is_exn h) = length (handler_variables h).
Proof.
  destruct h as [params metadata handler]. cbn [local_handler]. intros Hloc.
  apply andb_true_iff in Hloc as [Hloc _].
  apply andb_true_iff in Hloc as [Hlen Hparams]. apply Nat.eqb_eq in Hlen.
  pose proof (with_kinds_combine params (hm_params_kinds metadata) Hlen)
    as Hbound.
  set (bound := with_kinds params (hm_params_kinds metadata)) in *.
  set (mask := handler_mask sol is_exn (Handler params metadata handler))
    in *.
  assert (Hmlen : length mask = length bound)
    by (unfold mask, handler_mask; apply params_to_keep_length).
  assert (Hfst : map fst bound = params)
    by (rewrite Hbound; apply map_fst_combine, Hlen).
  rewrite <- Hbound in Hparams.
  destruct (params_rel_handler bound mask Hmlen Hparams) as [_ Hmatch].
  rewrite Hfst in Hmatch. cbn [handler_variables].
  split; [ exact Hmatch | ].
  rewrite Hmlen. unfold bound. apply with_kinds_length.
Qed.

Lemma conts_match_rec (kinds : finite_map flambda_kind)
    (C0 : finite_map (list (variable * bool)))
    (hs : list (continuation * continuation_handler)) :
  (∀ (k : continuation) (h : continuation_handler),
     (k, h) ∈ hs ->
     local_handler unboxed sol kinds C0 (handler_mask sol false h) h = true) ->
  conts_match
    (map (fun '(k, h) =>
            (k, combine (handler_variables h) (handler_mask sol false h)))
       hs)
    (map (fun '(k, h) => (k, handler_decisions false sol unboxed false h)) hs).
Proof.
  induction hs as [ | [k h] hs IHhs]; intros Hhs; [ constructor | ].
  cbn [map]. constructor.
  - split; [ reflexivity | ].
    exact (proj1 (handler_match _ _ false h (Hhs k h (or_introl eq_refl)))).
  - apply IHhs. intros k' h' Hin. apply (Hhs k' h'). right. exact Hin.
Qed.

Lemma lets_of_pairs (pairs : list (variable * variable * flambda_kind)) :
  ∀ body : expr,
    List.fold_left
      (fun body '(x, x', kind) => Let x kind (Simple (Var x')) body)
      pairs body
    = lets (map (fun '(x, x', kind) => (x, kind, Var x')) pairs) body.
Proof.
  induction pairs as [ | [[x x'] kind] pairs IH]; intros body;
    [ reflexivity | ].
  simpl. rewrite IH. reflexivity.
Qed.

(* The fields of a variable unboxed alike: an alias, the field of a
   block unboxed in turn, or the load of such a field. *)
Lemma bind_fields_pairs (fields fields_y : unboxed_fields) (body : expr) :
  fields_alike fields fields_y = true ->
  ∃ pairs : list (variable * variable * flambda_kind),
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs
    ∧ bind_fields (Unboxed fields) (Unboxed fields_y) body
      = Some (lets (pairs_lets pairs) body).
Proof.
  unfold fields_alike. intros Hf.
  destruct (unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y))
    as [pairs | ] eqn:Hp; [ | discriminate Hf ].
  exists pairs. split; [ reflexivity | ].
  unfold bind_fields. rewrite Hp, lets_of_pairs. reflexivity.
Qed.

Lemma lets_app (l l' : list (variable * flambda_kind * simple))
    (body : expr) :
  lets (l ++ l') body = lets l' (lets l body).
Proof. unfold lets. apply fold_left_app. Qed.

(* The fields of a block that is unboxed, some unboxed in turn. *)
Lemma bind_block_fields_lets2 (en : rebuild_env) (x : variable) (tag : Z)
    (args : list simple) (fields : unboxed_fields) :
  (∀ (f : field) (u : unboxed_field),
     (f, u) ∈ fields ->
     ∃ (i : nat) (a : simple),
       f = Field_block i Value ∧ nth_error args i = Some a
       ∧ s_field_used sol (Node_variable x) f = true
       ∧ kept_simple en a = true ∧ simple_kind en a = Value
       ∧ (((∃ v : variable, u = Not_unboxed v)
           ∧ plain_simple unboxed a = true)
          ∨ ∃ (nested : unboxed_fields) (b : variable)
              (fields_b : unboxed_fields),
              u = Unboxed nested ∧ a = Var b ∧ x <> b
              ∧ fields_of unboxed b = Some fields_b
              ∧ fields_alike nested fields_b = true)) ->
  ∀ body : expr,
    ∃ ls : list (list (variable * flambda_kind * simple)),
      bind_block_fields unboxed en tag args fields body
      = Some (lets (concat ls) body)
      ∧ Forall2
          (alloc_entry unboxed x
             (Prim (Variadic (Make_block (Block_kind_values tag)))
                (rebuild_fields sol en (Node_variable x) 0 args)))
          fields ls.
Proof.
  intros Hfields. unfold bind_block_fields.
  induction fields as [ | [f u] fields IH]; intros body.
  - exists []. split; [ reflexivity | constructor ].
  - destruct (Hfields f u (or_introl eq_refl))
      as (i & a & -> & Hnth & Hused & Hkept & Hkind & Hcase).
    assert (Hfield : nth_error (rebuild_fields sol en (Node_variable x) 0 args)
                       i
                     = Some a).
    { rewrite (rebuild_fields_nth en _ args 0 i a Hnth). simpl.
      unfold field_used. rewrite Hused, (kept_rewrite en a Hkept).
      reflexivity. }
    destruct Hcase as [((v & ->) & Ha)
                      | (nested & b & fb & -> & -> & Hxb & Hb & Halike)].
    + destruct (IH (fun f' u' Hin => Hfields f' u' (or_intror Hin))
                  (Let v Value (Simple a) body)) as (ls & Hl & Hentries).
      exists ([(v, Value, a)] :: ls). split.
      * simpl List.fold_left. rewrite Hnth, Hkind, kind_eqb_refl.
        rewrite (plain_not_bound a Ha). rewrite Hl.
        cbn [concat]. rewrite lets_app. reflexivity.
      * constructor; [ | exact Hentries ].
        cbn [alloc_entry]. exists Value, a.
        split; [ reflexivity | split; [ | exact Ha ] ].
        exists i.
        split; [ reflexivity | split; [ reflexivity | exact Hfield ] ].
    + destruct (bind_fields_pairs nested fb body Halike)
        as (pairs & Hp & Hl).
      destruct (IH (fun f' u' Hin => Hfields f' u' (or_intror Hin))
                  (lets (pairs_lets pairs) body)) as (ls & Hls & Hentries).
      exists (pairs_lets pairs :: ls). split.
      * simpl List.fold_left. rewrite Hnth, Hkind, kind_eqb_refl.
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable b))
          with (fields_of unboxed b).
        rewrite Hb, Hl, Hls. cbn [concat]. rewrite lets_app. reflexivity.
      * constructor; [ | exact Hentries ].
        cbn [alloc_entry]. exists b, fb, pairs.
        split; [ | split; [ exact Hb | split; [ exact Hxb | ] ] ];
          [ | split; [ exact Hp | reflexivity ] ].
        exists i.
        split; [ reflexivity | split; [ reflexivity | exact Hfield ] ].
Qed.

Definition ub_result (K : finite_map (list (option unboxed_fields)))
    (p q : expr) : Prop :=
  ub unboxed K p q
  ∧ free_continuations_of_expr p = free_continuations_of_expr q
  ∧ binds_statics p = binds_statics q.

Lemma ub_result_invalid (K : finite_map (list (option unboxed_fields))) :
  ub_result K Invalid Invalid.
Proof. split; [ constructor | split; reflexivity ]. Qed.

Lemma ac_rel_conts (K : finite_map (list (option unboxed_fields)))
    (ac ac' : apply_cont) :
  ac_rel unboxed K ac ac' ->
  apply_cont_continuations ac' = apply_cont_continuations ac.
Proof.
  intros (Hk & Htrap & _). unfold apply_cont_continuations.
  rewrite Hk, Htrap. reflexivity.
Qed.

Lemma pen_bind (en : unbox_env) (C : finite_map (list (variable * bool)))
    (bound : list (variable * flambda_kind)) :
  pen_of (ubind_variables en bound) C = bind_variables (pen_of en C) bound.
Proof. reflexivity. Qed.

Lemma rebuild_let_u_plain (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  named_plain unboxed d = true ->
  rebuild_let_u sol unboxed [] en x kind d body
  = Let x kind (rebuild_named sol en x d) body.
Proof.
  intros Hd. unfold rebuild_let_u.
  destruct (field_read d) as [[f arg] | ] eqn:Hread.
  - assert (Harg : plain_simple unboxed arg = true).
    { destruct d as [s0 | p args | | | | | ]; simpl in Hd; try discriminate Hd;
        [ discriminate Hread | ].
      destruct p as [p | p | p | p | p]; simpl in Hread; try discriminate Hread.
      destruct p; simpl in Hread; try discriminate Hread;
        destruct args as [ | a [ | ]]; simpl in Hread; try discriminate Hread;
        try (destruct variant_only; simpl in Hread; try discriminate Hread);
        injection Hread as _ <-; simpl in Hd; rewrite andb_true_r in Hd;
        exact Hd. }
    rewrite (plain_not_bound arg Harg).
    assert (Hch : changed_of [] arg = None) by (destruct arg; reflexivity).
    rewrite Hch, (named_plain_u en x d Hd). reflexivity.
  - rewrite (named_plain_u en x d Hd). reflexivity.
Qed.

Lemma plain_rebuild_let_static (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  plain_rebuild en (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding F sol preserve_direct_calls unit_code_ids
            first_fresh en)
         bindings)
      (plain_rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma unbox_rebuild_let_static (en : unbox_env)
    (bindings : list static_binding) (body : expr) :
  unbox_rebuild en (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding_u false false F sol cannot_change code_deps
            preserve_direct_calls unit_code_ids codes unboxed first_fresh_u []
            en)
         bindings)
      (unbox_rebuild (ubind_variables en []) body).
Proof. reflexivity. Qed.

Lemma plain_rebuild_let_cont (en : rebuild_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  plain_rebuild en (Let_cont_non_recursive k h body)
  = let mask := handler_mask sol (handler_is_exn_handler h) h in
    let body := plain_rebuild (bind_continuations en [(k, mask)]) body in
    if among k (free_continuations_of_expr body)
    then
      Let_cont_non_recursive k
        (rebuild_handler F sol preserve_direct_calls unit_code_ids
           first_fresh en true mask h)
        body
    else body.
Proof. reflexivity. Qed.

Lemma unbox_rebuild_let_cont (en : unbox_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  unbox_rebuild en (Let_cont_non_recursive k h body)
  = let decisions :=
      handler_decisions false sol unboxed (handler_is_exn_handler h) h in
    let body :=
      unbox_rebuild (ubind_continuations en [(k, decisions)]) body in
    if among k (free_continuations_of_expr body)
    then
      Let_cont_non_recursive k
        (rebuild_handler_u false false F sol cannot_change code_deps
           preserve_direct_calls unit_code_ids codes unboxed first_fresh_u []
           en true decisions h)
        body
    else body.
Proof. reflexivity. Qed.

Lemma plain_rebuild_let_cont_rec (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  plain_rebuild en (Let_cont_recursive handlers body)
  = let en :=
      bind_continuations en
        (map (fun '(k, h) => (k, handler_mask sol false h)) handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             rebuild_handler F sol preserve_direct_calls unit_code_ids
               first_fresh en false (handler_mask sol false h) h))
         handlers)
      (plain_rebuild en body).
Proof. reflexivity. Qed.

Lemma unbox_rebuild_let_cont_rec (en : unbox_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  unbox_rebuild en (Let_cont_recursive handlers body)
  = let en :=
      ubind_continuations en
        (map (fun '(k, h) => (k, handler_decisions false sol unboxed false h))
           handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k,
             rebuild_handler_u false false F sol cannot_change code_deps
               preserve_direct_calls unit_code_ids codes unboxed
               first_fresh_u [] en false
               (handler_decisions false sol unboxed false h) h))
         handlers)
      (unbox_rebuild en body).
Proof. reflexivity. Qed.

Lemma plain_rebuild_handler (en : rebuild_env) (dead : bool)
    (mask : list bool) (params : list variable) (metadata : handler_metadata)
    (handler : expr) :
  rebuild_handler F sol preserve_direct_calls unit_code_ids first_fresh en
    dead mask (Handler params metadata handler)
  = let bound := with_kinds params (hm_params_kinds metadata) in
    Handler (filter_kept mask params)
      (with_params_kinds metadata (filter_kept mask (hm_params_kinds metadata)))
      (if dead && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else plain_rebuild (bind_variables en bound) handler).
Proof. reflexivity. Qed.

Lemma unbox_rebuild_handler (en : unbox_env) (dead : bool)
    (decisions : list decision) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  rebuild_handler_u false false F sol cannot_change code_deps
    preserve_direct_calls unit_code_ids codes unboxed first_fresh_u [] en dead
    decisions
    (Handler params metadata handler)
  = let bound := with_kinds params (hm_params_kinds metadata) in
    let params := params_of_decisions bound decisions in
    Handler (map fst params) (with_params_kinds metadata (map snd params))
      (if dead && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else unbox_rebuild (ubind_variables en bound) handler).
Proof. reflexivity. Qed.

Lemma every_in {A : Type} (P : A -> Prop) (l : list A) (a : A) :
  every P l -> a ∈ l -> P a.
Proof.
  induction l as [ | b l IH]; intros Hl Ha; [ destruct Ha | ].
  destruct Hl as [Hb Hl]. destruct Ha as [<- | Ha];
    [ exact Hb | exact (IH Hl Ha) ].
Qed.

Lemma local_bindings_rebuilt (en : unbox_env)
    (C : finite_map (list (variable * bool)))
    (bindings : list static_binding) :
  forallb (local_binding unboxed) bindings = true ->
  flat_map
    (rebuild_binding_u false false F sol cannot_change code_deps
       preserve_direct_calls unit_code_ids codes unboxed first_fresh_u [] en)
    bindings
  = flat_map
      (rebuild_binding F sol preserve_direct_calls unit_code_ids first_fresh
         (pen_of en C))
      bindings
  ∧ forallb (local_binding unboxed)
      (flat_map
         (rebuild_binding F sol preserve_direct_calls unit_code_ids
            first_fresh (pen_of en C))
         bindings)
    = true.
Proof.
  induction bindings as [ | binding bindings IH]; simpl;
    [ split; reflexivity | ].
  rewrite andb_true_iff. intros [Hb Hbs]. destruct (IH Hbs) as [Heq Hplain].
  destruct binding as [id c | id | sym [ | | tag fields | | | ]];
    simpl in Hb; try discriminate Hb.
  simpl. rewrite Heq.
  destruct (is_symbol_used sol sym);
    simpl; [ | split; [ reflexivity | exact Hplain ] ].
  rewrite (rebuild_fields_u (ue_env en) (Node_symbol sym) fields Hb 0).
  rewrite (rebuild_fields_kinds (pen_of en C) (ue_env en) (Node_symbol sym)
             fields eq_refl 0).
  split; [ reflexivity | ]. rewrite Hplain, andb_true_r.
  apply rebuild_fields_plain, Hb.
Qed.

Lemma rebuild_ub :
  (∀ e : expr,
     ∀ (en : unbox_env) (C : finite_map (list (variable * bool)))
       (ce : finite_map (list node)),
       uinv en -> conts_match C (ue_decisions en) ->
       local_expr unboxed sol (env_kinds (ue_env en)) C e = true ->
       edges ce e ->
       ub_result (K_of C) (plain_rebuild (pen_of en C) e)
         (unbox_rebuild en e))
  ∧ (∀ h : continuation_handler,
       ∀ (en : unbox_env) (C : finite_map (list (variable * bool)))
         (ce : finite_map (list node)) (is_exn : bool),
         uinv en -> conts_match C (ue_decisions en) ->
         local_handler unboxed sol (env_kinds (ue_env en)) C
           (handler_mask sol is_exn h) h
         = true ->
         FlambdaReaperFlowProofs.handler_edges (graph_of_facts F) noted deps
           cds kept_slots codes0 symbols True False None ce h ->
         ∀ dead : bool,
           ∃ (ps ps' : list (variable * flambda_kind))
             (metadata metadata' : handler_metadata) (hp hu : expr),
             rebuild_handler F sol preserve_direct_calls unit_code_ids
               first_fresh (pen_of en C) dead (handler_mask sol is_exn h) h
             = Handler (map fst ps) metadata hp
             ∧ rebuild_handler_u false false F sol cannot_change code_deps
                 preserve_direct_calls unit_code_ids codes unboxed
                 first_fresh_u [] en dead
                 (handler_decisions false sol unboxed is_exn h) h
               = Handler (map fst ps') metadata' hu
             ∧ params_rel unboxed
                 (flat_map shape_of_param
                    (combine (handler_variables h)
                       (handler_mask sol is_exn h)))
                 ps ps'
             ∧ hm_params_kinds metadata = map snd ps
             ∧ hm_params_kinds metadata' = map snd ps'
             ∧ Forall2 decision_match
                 (combine (handler_variables h) (handler_mask sol is_exn h))
                 (handler_decisions false sol unboxed is_exn h)
             ∧ length (handler_mask sol is_exn h)
               = length (handler_variables h)
             ∧ ub_result (K_of C) hp hu).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hxf Hbind]. apply negb_true_iff in Hxf.
    destruct Hedges as [Hnamed_edges Hbody_edges].
    specialize (IH (ubind_variables en [(x, kind)]) C ce
                  (uinv_bind en [(x, kind)]) HC Hbody Hbody_edges).
    rewrite pen_bind in IH.
    unfold plain_rebuild, unbox_rebuild in *. cbn [rebuild_expr rebuild_expr_u].
    unfold is_pending_projection.
    rewrite (uinv_projections en Hen), pen_projections.
    rewrite (rebuild_named_kinds (pen_of en C) (ue_env en) x d eq_refl).
    assert (Hd : (∃ s, d = Simple s) ∨ (∃ p args, d = Prim p args)).
    { destruct (fields_of unboxed x) as [fields | ] eqn:Hx.
      - destruct (unboxed_binding_cases _ x kind d fields Hbind)
          as (_ & [(tag & args & -> & _) | [(bn & c & -> & _)
                  | [(y & fy & -> & _)
                    | (bk & mut & i & y & fy & n & -> & _)]]]);
          [ right | right | left | right ]; eauto.
      - destruct (plain_binding_cases x kind d Hbind)
          as [Hp | (p & y & f & fs & -> & _)]; [ | right; eauto ].
        destruct d; try discriminate Hp; eauto. }
    destruct IH as (IHub & IHfc & IHbs).
    destruct Hd as [(s0 & ->) | (p & args & ->)]; cbn beta iota;
      (destruct (is_dead_var sol x kind) eqn:Hdead;
       [ exact (ub_result_invalid _) | ]);
      (destruct (is_var_used sol x kind) eqn:Hused;
       [ | split; [ exact IHub | split; assumption ] ]);
      change (unboxed_of_node unboxed (Node_variable x))
        with (fields_of unboxed x);
      change (changed_of_node [] (Node_variable x)) with (@None closure_repr);
      cbn iota;
      (destruct (fields_of unboxed x) as [fields | ] eqn:Hx).
    + (* An alias of an unboxed variable. *)
      destruct (unboxed_binding_cases _ x kind _ fields Hbind)
        as (-> & [(tag & args & Heq & _) | [(bn & c & Heq & _)
                 | [(y & fields_y & Heq & Hne & Hy & Hflat & Hyused)
                   | (bk & mut & i & y & fy & n & Heq & _)]]]);
        try discriminate Heq.
      injection Heq as ->.
      cbn [rebuild_named rewrite_simple].
      unfold kind_of_variable in Hyused |- *.
      cbn [env_kinds kinds_env] in Hyused. rewrite Hyused.
      unfold rebuild_unboxed_binding.
      destruct (bind_fields_pairs fields fields_y
                  (rebuild_expr_u false false F sol cannot_change code_deps
                     preserve_direct_calls unit_code_ids codes unboxed
                     first_fresh_u [] (ubind_variables en [(x, Value)]) body)
                  Hflat)
        as (pairs & Hentries & Hl).
      cbn [unboxed_of].
      change (unboxed_of_node unboxed (Node_variable y))
        with (fields_of unboxed y).
      rewrite Hy, Hl.
      destruct (lets_free_continuations (pairs_lets pairs)
                  (rebuild_expr_u false false F sol cannot_change code_deps
                     preserve_direct_calls unit_code_ids codes unboxed
                     first_fresh_u [] (ubind_variables en [(x, Value)]) body))
        as [Hfc Hbs].
      split; [ exact (ub_alias unboxed _ x y fields fields_y pairs _ _ Hx Hxf
                        Hy Hne Hentries IHub)
             | ].
      cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
      split; assumption.
    + (* A simple. *)
      destruct (plain_binding_cases x kind _ Hbind)
        as [Hp | (p & y & f & fs & Heq & _)]; [ | discriminate Heq ].
      rewrite (rebuild_let_u_plain _ x kind _ _ Hp).
      split; [ | split; assumption ].
      constructor; [ | apply named_plain_rebuilt, Hp | | exact IHub ].
      { apply plain_variable_iff. unfold Keyed.is_bound.
        unfold fields_of in Hx. rewrite Hx.
        split; [ reflexivity | exact Hxf ]. }
      apply rebuild_named_not_heap.
      eapply FlambdaReaperFlowProofs.named_edges_not_heap.
      exact Hnamed_edges.
    + (* An unboxed allocation, or the load of a field unboxed in turn. *)
      destruct (unboxed_binding_cases _ x kind _ fields Hbind)
        as (-> & [(tag & args' & Heq & Hfields)
                 | [(bn & c & Heq & Hc & Hfields)
                   | [(y & fy & Heq & _)
                     | (bk & mut & i & y & fy & nested & Heq & Hne & Hy & Hny
                        & Halike)]]]);
        [ | | discriminate Heq | ];
        injection Heq as -> ->; simpl in Hnamed_edges;
        destruct Hnamed_edges as [_ Hnamed_edges].
      * destruct (bind_block_fields_lets2 (ue_env en) x tag args' fields)
          with (body := rebuild_expr_u false false F sol cannot_change code_deps
                          preserve_direct_calls unit_code_ids codes unboxed
                          first_fresh_u [] (ubind_variables en [(x, Value)])
                          body)
          as (ls & Hl & Hentries).
        { intros f u Hin.
          destruct (Hfields f u Hin) as (i & a & -> & Hnth & Hkind & Hcase).
          rewrite kinds_env_kind in Hkind.
          destruct (field_kept x fields _ u a (ue_env en) Hx Hin
                      (proj1 Hnamed_edges i a Hnth) eq_refl)
            as [Hused' Hkept].
          exists i, a. repeat split; assumption. }
        unfold rebuild_unboxed_binding. rewrite Hl. cbn [rebuild_named].
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u false false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split; [ refine (ub_alloc unboxed _ x _ fields ls _ _ Hx Hxf Hentries
                           _ IHub);
                 intros q0 args0 Hq0; discriminate Hq0
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; assumption.
      * destruct (bind_boxed_number_lets x bn c fields
                    (rebuild_named sol (ue_env en) x
                       (Prim (Unary (Box_number bn)) [c])) Hc)
          with (body := rebuild_expr_u false false F sol cannot_change code_deps
                          preserve_direct_calls unit_code_ids codes unboxed
                          first_fresh_u [] (ubind_variables en [(x, Value)])
                          body)
          as (ls & Hl & Hentries).
        { intros Hne. destruct fields as [ | [f u] fields]; [ easy | ].
          destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
          destruct (field_kept x _ _ (Not_unboxed v) c (ue_env en) Hx
                      (or_introl eq_refl) (proj1 Hnamed_edges) eq_refl)
            as [Hused' Hkept].
          cbn [rebuild_named].
          unfold field_used. rewrite Hused'. simpl.
          rewrite (kept_rewrite _ c Hkept). reflexivity. }
        { intros f u Hin. exact (Hfields f u Hin). }
        unfold rebuild_unboxed_binding. rewrite Hl.
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u false false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split; [ refine (ub_alloc unboxed _ x _ fields _ _ _ Hx Hxf Hentries
                           _ IHub);
                 intros q0 args0 Hq0; cbn [rebuild_named] in Hq0;
                 destruct (field_used _ _ _); discriminate Hq0
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; assumption.
      * (* The load of a field unboxed in turn. *)
        cbn [rebuild_named]. unfold rebuild_unboxed_binding, load_unboxed_field.
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hny.
        destruct (bind_fields_pairs fields nested
                    (rebuild_expr_u false false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body)
                    Halike)
          as (pairs & Hentries & Hl).
        rewrite Hl.
        destruct (lets_free_continuations (pairs_lets pairs)
                    (rebuild_expr_u false false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split.
        -- apply (ub_load unboxed _ x Value _ (Field_block (Z.to_nat i) Value)
                    (rewrite_simple sol (ue_env en) (Var y)) y fields fy
                    nested pairs); try assumption.
           ++ reflexivity.
           ++ simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
              right. eexists. reflexivity.
           ++ exact (keyed_lookup_in field_eqb field_eqb_decides _ _ fy Hny).
        -- cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
           split; assumption.
    + (* A primitive of a variable that is not unboxed. *)
      destruct (plain_binding_cases x kind _ Hbind)
        as [Hp | (p' & y & f & fs & Heq & Hread & Hreads & Hnamed & Hy
                  & Hfield)].
      * rewrite (rebuild_let_u_plain _ x kind _ _ Hp).
        split; [ | split; assumption ].
        constructor; [ | apply named_plain_rebuilt, Hp | | exact IHub ].
        { apply plain_variable_iff. unfold Keyed.is_bound.
          unfold fields_of in Hx. rewrite Hx.
          split; [ reflexivity | exact Hxf ]. }
        apply rebuild_named_not_heap.
        eapply FlambdaReaperFlowProofs.named_edges_not_heap.
        exact Hnamed_edges.
      * rewrite Heq in Hnamed, Hnamed_edges |- *.
        rewrite (Hnamed (ue_env en)).
        assert (Hacc : g_accessor (graph_of_facts F) (Node_variable x) f
                         (Node_variable y)).
        { clear -Hread Hnamed_edges.
          destruct p'; try discriminate Hread; simpl in Hread;
            injection Hread as <-; exact (proj1 (proj2 Hnamed_edges)). }
        destruct (reads_found x y kind f fs Hx Hy Hacc Hdead Hused Hfield)
          as (v & Hv).
        unfold rebuild_let_u.
        assert (Hfr : field_read (Prim (Unary p') [Var y]) = Some (f, Var y)).
        { clear -Hread. destruct p'; try discriminate Hread; exact Hread. }
        rewrite Hfr. cbn [unboxed_of]. change (unboxed_of_node unboxed
          (Node_variable y)) with (fields_of unboxed y).
        rewrite Hy, Hv.
        split; [ | split; assumption ].
        apply (ub_read unboxed _ x kind
                 (Prim (Unary p') [rewrite_simple sol (ue_env en) (Var y)]) f
                 (rewrite_simple sol (ue_env en) (Var y)) y v fs).
        -- apply plain_variable_iff. unfold Keyed.is_bound.
           unfold fields_of in Hx. rewrite Hx.
           split; [ reflexivity | exact Hxf ].
        -- apply Hreads.
        -- simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
           right. eexists. reflexivity.
        -- exact Hy.
        -- exact (keyed_lookup_in field_eqb field_eqb_decides f _ fs Hv).
        -- exact IHub.
  - (* Let_static *)
    intros bindings body _ IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    destruct (IH (ubind_variables en []) C ce (uinv_bind en []) HC Hbody
                (proj2 Hedges))
      as (IHub & IHfc & IHbs).
    rewrite pen_bind in IHub, IHfc, IHbs.
    destruct (local_bindings_rebuilt en C bindings Hbs) as [Heq Hplain].
    rewrite plain_rebuild_let_static, unbox_rebuild_let_static, Heq.
    split; [ constructor; [ exact Hplain | exact IHub ] | ].
    cbn [free_continuations_of_expr binds_statics]. rewrite IHfc, IHbs.
    split; reflexivity.
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hh Hbody].
    destruct Hedges as [Hh_edges Hbody_edges].
    rewrite plain_rebuild_let_cont, unbox_rebuild_let_cont. cbv zeta.
    destruct (IHh en C ce (handler_is_exn_handler h) Hen HC Hh Hh_edges true)
      as (ps & ps' & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd'
          & Hmatch & Hlen & Hhub & Hhfc & Hhbs).
    set (is_exn := handler_is_exn_handler h) in *.
    set (mask := handler_mask sol is_exn h) in *.
    set (ds := handler_decisions false sol unboxed is_exn h) in *.
    set (C' := (k, combine (handler_variables h) mask) :: C).
    assert (HC' : conts_match C' (ue_decisions
                                   (ubind_continuations en [(k, ds)]))).
    { constructor; [ split; [ reflexivity | exact Hmatch ] | exact HC ]. }
    assert (Hpen : pen_of (ubind_continuations en [(k, ds)]) C'
                   = bind_continuations (pen_of en C) [(k, mask)]).
    { unfold pen_of, C'. cbn [map env_conts env_kinds ue_env
        ubind_continuations bind_continuations app].
      rewrite map_snd_combine by exact (eq_sym Hlen). reflexivity. }
    destruct (IH (ubind_continuations en [(k, ds)]) C' _
                (uinv_continuations en [(k, ds)]) HC' Hbody Hbody_edges)
      as (IHub & IHfc & IHbs).
    rewrite Hpen in IHub, IHfc, IHbs.
    rewrite <- IHfc.
    destruct (among k _) eqn:Hk.
    + rewrite Hp, Hu.
      split; [ eapply ub_let_cont; eassumption | ].
      cbn [free_continuations_of_expr free_continuations_of_handler
           binds_statics handler_binds_statics].
      rewrite Hhfc, IHfc, Hhbs, IHbs. split; reflexivity.
    + split; [ | split; assumption ].
      apply (ub_agree unboxed _ _ _ IHub). intros k' Hk'.
      unfold C', K_of. cbn [map]. rewrite lookup_cons.
      destruct (Pos.eqb k' k) eqn:E; [ | reflexivity ].
      apply Pos.eqb_eq in E. subst k'. apply among_In in Hk'.
      rewrite Hk' in Hk. discriminate.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite forallb_forall in Hhs.
    destruct Hedges as [Hhs_edges Hbody_edges].
    rewrite plain_rebuild_let_cont_rec, unbox_rebuild_let_cont_rec. cbv zeta.
    set (dss := map (fun '(k, h) =>
                       (k, handler_decisions false sol unboxed false h))
                  handlers).
    set (masks := map (fun '(k, h) => (k, handler_mask sol false h)) handlers).
    set (CR := map (fun '(k, h) =>
                      (k, combine (handler_variables h)
                            (handler_mask sol false h)))
                 handlers).
    set (S := map (fun '(k, h) =>
                     (k, flat_map shape_of_param
                           (combine (handler_variables h)
                              (handler_mask sol false h))))
                handlers).
    set (C' := CR ++ C).
    assert (Hhs' : ∀ (k : continuation) (h : continuation_handler),
               (k, h) ∈ handlers ->
               local_handler unboxed sol (env_kinds (ue_env en)) C'
                 (handler_mask sol false h) h = true)
      by (intros k h Hin; exact (Hhs _ Hin)).
    clear Hhs.
    assert (HC' : conts_match C' (ue_decisions
                                   (ubind_continuations en dss))).
    { cbn [ue_decisions ubind_continuations]. apply Forall2_app; [ | exact HC ].
      exact (conts_match_rec _ C' handlers Hhs'). }
    assert (Hpen : pen_of (ubind_continuations en dss) C'
                   = bind_continuations (pen_of en C) masks).
    { unfold pen_of, C', bind_continuations.
      cbn [ue_env ubind_continuations env_conts env_kinds].
      rewrite map_app.
      match goal with
      | |- context [map ?f CR] => assert (Hm : map f CR = masks)
      end.
      { unfold CR, masks. rewrite map_map.
        apply map_ext_in. intros [k h] Hin. rewrite map_snd_combine;
          [ reflexivity | ].
        symmetry. exact (proj2 (handler_match _ _ false h (Hhs' k h Hin))). }
      f_equal; try reflexivity. rewrite <- Hm. reflexivity. }
    assert (HK : K_of C' = S ++ K_of C).
    { unfold C', K_of, CR, S. rewrite map_app, map_map. f_equal.
      apply map_ext. intros [k h]. reflexivity. }
    destruct (IH (ubind_continuations en dss) C' _
                (uinv_continuations en dss) HC' Hbody Hbody_edges)
      as (IHub & IHfc & IHbs).
    rewrite Hpen in IHub, IHfc, IHbs.
    assert (Hhandlers : ∀ hs : list (continuation * continuation_handler),
      (∀ (k : continuation) (h : continuation_handler),
         (k, h) ∈ hs -> (k, h) ∈ handlers) ->
      let hps :=
        map
          (fun '(k, h) =>
             (k,
              rebuild_handler F sol preserve_direct_calls unit_code_ids
                first_fresh (bind_continuations (pen_of en C) masks) false
                (handler_mask sol false h) h))
          hs
      in
      let hus :=
        map
          (fun '(k, h) =>
             (k,
              rebuild_handler_u false false F sol cannot_change code_deps
                preserve_direct_calls unit_code_ids codes unboxed
                first_fresh_u [] (ubind_continuations en dss) false
                (handler_decisions false sol unboxed false h) h))
          hs
      in
      ub_handlers unboxed (K_of C') hps hus
        (map (fun '(k, h) =>
                (k, flat_map shape_of_param
                      (combine (handler_variables h)
                         (handler_mask sol false h))))
           hs)
      ∧ handlers_bind_statics hps = handlers_bind_statics hus
      ∧ flat_map (fun '(_, h) => free_continuations_of_handler h) hps
        = flat_map (fun '(_, h) => free_continuations_of_handler h) hus
      ∧ map fst hps = map fst hus).
    { induction hs as [ | [k h] hs IHhs]; intros Hsub; cbv zeta;
        [ cbn [map]; split; [ apply ub_handlers_nil | ];
          split; [ reflexivity | split; reflexivity ] | ].
      destruct IHhs as (Hub_hs & Hbs_hs & Hfc_hs & Hfst_hs).
      { intros k' h' Hin. apply Hsub. right. exact Hin. }
      assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
      destruct (IHh k h Hin (ubind_continuations en dss) C' _ false
                  (uinv_continuations en dss) HC' (Hhs' _ _ Hin)
                  (every_in _ _ (k, h) Hhs_edges Hin) false)
        as (ps & ps' & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd' & _
            & _ & Hhub & Hhfc & Hhbs).
      rewrite Hpen in Hp. cbv zeta. cbn [map]. rewrite Hp, Hu.
      split; [ apply ub_handlers_cons; assumption | ].
      unfold handlers_bind_statics in *. cbn [existsb flat_map map fst].
      cbn [free_continuations_of_handler handler_binds_statics].
      rewrite Hhbs, Hbs_hs, Hhfc, Hfc_hs, Hfst_hs. split; [ | split ];
        reflexivity. }
    destruct (Hhandlers handlers (fun k h Hin => Hin))
      as (Hub_hs & Hbs_hs & Hfc_hs & Hfst_hs).
    fold S in Hub_hs.
    rewrite HK in Hub_hs, IHub.
    split; [ apply (ub_let_cont_rec unboxed (K_of C) S); assumption | ].
    unfold handlers_bind_statics in Hbs_hs.
    cbn [free_continuations_of_expr binds_statics].
    rewrite Hfst_hs, Hfc_hs, IHfc, IHbs, Hbs_hs. split; reflexivity.
  - (* Apply *)
    intros ap en C ce Hen HC Hloc. discriminate Hloc.
  - (* Apply_cont *)
    intros ac en C ce Hen HC Hloc _. unfold plain_rebuild, unbox_rebuild.
    cbn [rebuild_expr rebuild_expr_u]. simpl in Hloc.
    unfold rebuild_apply_cont, rebuild_apply_cont_u.
    destruct (rewrite_apply_cont_rel en C ac HC Hloc)
      as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
      rewrite Hp, Hu; [ exact (ub_result_invalid _) | ].
    split; [ constructor; exact Hrel | ].
    cbn [free_continuations_of_expr binds_statics].
    rewrite (ac_rel_conts _ _ _ Hrel). split; reflexivity.
  - (* Switch *)
    intros scrutinee arms en C ce Hen HC Hloc _. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hs Harms].
    unfold plain_rebuild, unbox_rebuild. cbn [rebuild_expr rebuild_expr_u].
    unfold rebuild_switch, rebuild_switch_u.
    assert (Hcompact : ∀ (d : Z) (ac : apply_cont),
               (d, ac) ∈ Keyed.compact Z.eqb arms ->
               apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true).
    { intros d ac Hin.
      apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hin.
      apply (keyed_lookup_in Z.eqb z_eqb_decides_equality) in Hin.
      rewrite forallb_forall in Harms. exact (Harms _ Hin). }
    assert (Harms' :
      ∀ l : list (Z * apply_cont),
        (∀ (d : Z) (ac : apply_cont), (d, ac) ∈ l ->
           apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true) ->
        Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm ∧ ac_rel unboxed (K_of C) (snd arm) (snd arm'))
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont sol (pen_of en C) ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol unboxed en ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)).
    { induction l as [ | [d ac] l IHl]; intros Hl; [ constructor | ].
      assert (Hrest : Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm ∧ ac_rel unboxed (K_of C) (snd arm) (snd arm'))
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont sol (pen_of en C) ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol unboxed en ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)).
      { apply IHl. intros d' ac' Hin. apply (Hl d' ac'). right. exact Hin. }
      cbn [flat_map].
      destruct (rewrite_apply_cont_rel en C ac HC (Hl d ac (or_introl eq_refl)))
        as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
        rewrite Hp, Hu; [ exact Hrest | ].
      constructor; [ split; [ reflexivity | exact Hrel ] | exact Hrest ]. }
    pose proof (Harms' _ Hcompact) as Hrel.
    rewrite (rewrite_simple_kinds (pen_of en C) (ue_env en) scrutinee eq_refl).
    rewrite (rewrite_simple_u _ scrutinee Hs).
    destruct Hrel as [ | arm arm' l l' Harm Hrel ];
      [ exact (ub_result_invalid _) | ].
    cbn [Keyed.is_empty].
    split; [ apply ub_switch; [ apply rewrite_simple_plain, Hs | ] | ].
    { constructor; assumption. }
    split; [ | reflexivity ].
    cbn [free_continuations_of_expr flat_map].
    destruct Harm as [_ Hac]. rewrite (ac_rel_conts _ _ _ Hac). f_equal.
    clear -Hrel. induction Hrel as [ | a a' l l' [_ Ha] _ IH ];
      [ reflexivity | ].
    cbn [flat_map]. rewrite (ac_rel_conts _ _ _ Ha), IH. reflexivity.
  - (* Invalid *)
    intros en C ce Hen HC Hloc _. exact (ub_result_invalid _).
  - (* Handler *)
    intros params metadata handler IH en C ce is_exn Hen HC Hloc Hedges dead.
    simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hlen Hparams]. apply Nat.eqb_eq in Hlen.
    pose proof (with_kinds_combine params (hm_params_kinds metadata) Hlen)
      as Hbound.
    set (bound := with_kinds params (hm_params_kinds metadata)) in *.
    set (mask := handler_mask sol is_exn (Handler params metadata handler))
      in *.
    assert (Hmlen : length mask = length bound)
      by (unfold mask, handler_mask; apply params_to_keep_length).
    assert (Hfst : map fst bound = params)
      by (rewrite Hbound; apply map_fst_combine, Hlen).
    assert (Hsnd : map snd bound = hm_params_kinds metadata)
      by (rewrite Hbound; apply map_snd_combine, Hlen).
    rewrite <- Hbound in Hparams.
    destruct (params_rel_handler bound mask Hmlen Hparams) as [Hrel Hmatch].
    rewrite Hfst in Hrel, Hmatch.
    assert (Hbody' : local_expr unboxed sol
                       (env_kinds (ue_env (ubind_variables en bound))) C
                       handler = true).
    { cbn [ue_env ubind_variables ubind_variables_then ubind_set
           bind_variables_then env_kinds].
      rewrite Hbound. exact Hbody. }
    destruct (IH _ C ce (uinv_bind en bound) HC Hbody' Hedges)
      as (IHub & IHfc & IHbs).
    rewrite pen_bind in IHub, IHfc, IHbs.
    rewrite plain_rebuild_handler, unbox_rebuild_handler. cbv zeta.
    fold bound.
    exists (filter_kept mask bound),
      (params_of_decisions bound
         (handler_decisions false sol unboxed is_exn
            (Handler params metadata handler))),
      (with_params_kinds metadata
         (filter_kept mask (hm_params_kinds metadata))),
      (with_params_kinds metadata
         (map snd
            (params_of_decisions bound
               (handler_decisions false sol unboxed is_exn
                  (Handler params metadata handler))))).
    eexists _, _.
    split; [ rewrite <- filter_kept_map, Hfst; reflexivity | ].
    split; [ reflexivity | ].
    split; [ exact Hrel | ].
    split; [ rewrite hm_with_params_kinds, <- Hsnd, filter_kept_map;
             reflexivity | ].
    split; [ apply hm_with_params_kinds | ].
    split; [ exact Hmatch | ].
    split; [ rewrite Hmlen; unfold bound; apply with_kinds_length | ].
    destruct (dead && _); [ exact (ub_result_invalid _) | ].
    split; [ exact IHub | split; assumption ].
Qed.

End Rebuilds.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma uinv_empty : uinv empty_unbox_env.
Proof. split; reflexivity. Qed.

Theorem rebuild_unbox_refines_rebuild :
  rebuild_unbox_refines_rebuild_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox
    preserve_direct_calls Hslots Hfragment Hclosed Hlocal Hcheck.
  set (F := traverse_unit flag u e) in *.
  unfold local_unboxing in Hlocal.
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  unfold rebuild_unit_unbox, rebuild_unit_unbox_repr,
    rebuild_unit_unbox_repr_mode, rebuild_unit_unbox_repr_modes.
  destruct (unit_unboxed_fields F T dominated to_unbox e) as [unboxed next]
    eqn:Hunboxed.
  cbn [fst] in Hlocal, Hdistinct, Hfresh.
  rename Hlocal into Hloc.
  pose proof (closure_free_reads_blocks flag u e
                (proj1 (local_closure_free unboxed (solution_of_tables T))
                   e [] [] Hloc))
    as Hreads.
  apply (proj1 (strong_local unboxed (solution_of_tables T)) e [] [] Hloc)
    in Hfresh.
  rename Hfresh into Hloc'.
  assert (Hchanged :
    unit_changed_representations (fun _ => false) F T dominated to_unbox [] e
    = []) by reflexivity.
  rewrite Hchanged.
  pose proof (check_closed_sound is_local F T Hclosed) as Hcl.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts flag u e True
              Hfragment (or_introl I))
    as (noted & deps & cds & Hedges & _).
  assert (Hfields_read :
    ∀ (x : variable) (fields : unboxed_fields) (f : field)
      (v : unboxed_field),
      fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
      ∃ w base : node,
        g_accessor (graph_of_facts F) w f base
        ∧ s_usages (solution_of_tables T) (Node_variable x) base = true
        ∧ s_has_usage (solution_of_tables T) w = true).
  { intros x fields f v Hx Hin. apply (fields_of_in unboxed) in Hx.
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox (fresh_name (names_of e))))
      by (rewrite Hunboxed; exact Hx).
    destruct (unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _ Hx')
      as (a & next' & ->).
    destruct (mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f _
                Hin)
      as (use & Huse).
    destruct (get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (add_usages_none F T _ Hreads) in Hbase.
    apply mem_node_In in Hbase.
    destruct (get_direct_usages_in T _ _ Hbase) as [Hus _].
    exists w, base. split; [ exact Hacc | split; [ exact Hus | ] ].
    destruct Hw as [Hw | Hw]; [ | exact Hw ].
    exact (has_usage_any _ _ _ Hcl _ Hw). }
  assert (Hreads_found :
    ∀ (x y : variable) (kind : flambda_kind) (f : field)
      (fields : unboxed_fields),
      fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
      g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
      is_dead_var (solution_of_tables T) x kind = false ->
      is_var_used (solution_of_tables T) x kind = true ->
      kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
      ∃ v : variable,
        Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
  { intros x y kind f fields Hx Hy Hacc Hdead Hused Hkind.
    assert (Hsx : mem_node (Node_variable x) (t_has_source T) = true).
    { unfold is_dead_var in Hdead.
      destruct Hkind as [-> | (nk & ->)]; simpl in Hdead;
        apply negb_false_iff in Hdead; exact Hdead. }
    assert (Hux : mem_node (Node_variable x) (t_has_usage T) = true).
    { unfold is_var_used in Hused.
      destruct Hkind as [-> | (nk & ->)]; exact Hused. }
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hhx : has_to_be_unboxed dominated to_unbox (Node_variable x)
                  = false).
    { destruct (has_to_be_unboxed dominated to_unbox (Node_variable x))
        eqn:Hh; [ | reflexivity ].
      unfold has_to_be_unboxed in Hh.
      apply existsb_exists in Hh as ([a x'] & Hin & Htest).
      apply andb_true_iff in Htest as [Hx' Ha].
      apply node_eqb_eq in Hx'. subst x'.
      pose proof (unboxed_fields_of_names_bound F T
                    (bound_code_my_closures e) dominated to_unbox
                    (fresh_name (names_of e)) a (Node_variable x) Hin Ha)
        as Hb.
      rewrite Hunboxed in Hb. cbn [fst] in Hb. unfold fields_of in Hx.
      exfalso.
      exact (keyed_lookup_none_not_in node_eqb node_eqb_decides _ _ Hx Hb). }
    apply (fields_of_in unboxed) in Hy.
    assert (Hy' : (Node_variable y, fields)
                  ∈ fst (unboxed_fields_of_names F T
                           (bound_code_my_closures e) dominated to_unbox
                           (fresh_name (names_of e))))
      by (rewrite Hunboxed; exact Hy).
    destruct (unboxed_fields_of_names_dom F T _ dominated to_unbox _ _ _ Hy')
      as (a & next' & Hdom & Ha & ->).
    unfold check_unboxed_reads in Hcheck. rewrite forallb_forall in Hcheck.
    specialize (Hcheck _ Hacc). cbn beta iota in Hcheck.
    rewrite Hsx, Hux, Hhx in Hcheck. simpl in Hcheck.
    rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hdom).
    cbn beta iota in Hcheck. rewrite node_eqb_refl, Ha in Hcheck.
    simpl in Hcheck. apply andb_true_iff in Hcheck as [Hnau Hsrc].
    apply negb_true_iff in Hnau.
    assert (Hblock : is_block_like_field f = true).
    { rewrite forallb_forall in Hreads. exact (Hreads _ Hacc). }
    assert (Hfs : is_function_slot f = false)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hdata : is_data_field f = true)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hus : mem_pair (Node_variable y) (Node_variable y) (t_usages T)
                  = true)
      by exact (usages_accessor _ _ _ Hcl _ _ _ Hacc Hnau).
    assert (Hhu : mem_node (Node_variable y) (t_has_usage T) = true)
      by exact (has_usage_accessor _ _ _ Hcl _ _ _ Hux Hacc).
    assert (Huses : mem_node (Node_variable y)
                      (add_usages_through_function_slots F T
                         (bound_code_my_closures e) true
                         (get_direct_usages T [Node_variable y]))
                    = true).
    { rewrite (add_usages_none F T _ Hreads). apply mem_node_In.
      apply get_direct_usages_self; assumption. }
    destruct (get_fields_found F T _ _ _ _ Hacc Huses Hfs Hux)
      as [Hex Hall].
    apply (mk_unboxed_fields_lookup F T _ dominated to_unbox _ a f Hdata).
    - intros Hno. rewrite Hno in Hsrc. discriminate Hsrc.
    - exact Hex.
    - intros use Hin.
      destruct (Hall use Hin) as [-> | (flow_to & -> & Hw)];
        [ left; reflexivity | right ].
      exists flow_to. split; [ reflexivity | ].
      destruct (forallb _ flow_to) eqn:Hft; [ | reflexivity ].
      rewrite forallb_forall in Hft. rewrite (Hft _ Hw) in Hhx.
      discriminate Hhx. }
  destruct (proj1 (rebuild_ub F (solution_of_tables T)
                     (fun n => mem_node n cannot_change) code_deps
                     preserve_direct_calls (static_names code_ids_bound e)
                     (bound_code_metadata e) unboxed
                     (fresh_name (names_of e)) next noted deps cds
                     (unit_kept e) (unit_codes e)
                     (static_names symbols_of_binding e) is_local Hcl Hslots
                     Hfields_read Hreads_found)
              e empty_unbox_env [] (unit_conts u) uinv_empty (Forall2_nil _)
              Hloc' Hedges)
    as (Hub & _ & _).
  exact (ub_refines unboxed Hdistinct _ _ Hub).
Qed.

(* Refinement up to observation, then refinement, is refinement up to
   observation. *)
Lemma refines_observably_refines (s m t : state) :
  refines_observably s m -> refines m t -> refines_observably s t.
Proof.
  intros (Hexit & Hback & Hwrong) (Hexit' & Hback').
  split; [ | split ].
  - intros k vs hp Hs. destruct (Hexit k vs hp Hs) as (vs' & Hm & Hvs).
    exists vs'. split; [ exact (Hexit' k vs' hp Hm) | exact Hvs ].
  - intros k vs' hp Ht.
    destruct (Hback' (Exits k vs' hp) ltac:(discriminate) Ht) as [Hm | Hm].
    + exact (Hback k vs' hp Hm).
    + right. exact (Hwrong Hm).
  - intros Ht. destruct (Hback' Goes_wrong ltac:(discriminate) Ht) as [Hm | Hm];
      exact (Hwrong Hm).
Qed.

Theorem rebuild_unbox_refines_observably_local :
  rebuild_unbox_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls Hlocal_slots Hfragment Hstatic Hkinded
    Hwell_formed Hone Hclosed Hlrs Hexact Hlocal Hcheck.
  apply (refines_observably_refines _
           (State (rebuild_unit_from_tables (traverse_unit flag u e) T
                     preserve_direct_calls e)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (rebuild_refines_observably_local flag is_local u e T
             preserve_direct_calls Hlocal_slots Hfragment
             (static_constants_any_kind_arguments _ e Hstatic)
             Hkinded Hwell_formed Hone Hclosed Hlrs Hexact).
  - exact (rebuild_unbox_refines_rebuild flag is_local u e T cannot_change
             (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls
             Hlocal_slots Hfragment Hclosed Hlocal Hcheck).
Qed.

Theorem rebuild_unbox_refines_observably :
  rebuild_unbox_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls Hlocal_slots Hprojections Hfragment Hstatic Hkinded
    Hwell_formed Hone Hclosed Hexact Hlocal Hcheck.
  exact (rebuild_unbox_refines_observably_local flag is_local u e T
           cannot_change dominated to_unbox preserve_direct_calls Hlocal_slots
           Hfragment Hstatic Hkinded Hwell_formed Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal_slots Hprojections)
           Hexact Hlocal Hcheck).
Qed.

Theorem local_example : local_example_statement.
Proof.
  split.
  { intros f Hf. discriminate Hf. }
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem fragment_examples : fragment_examples_statement.
Proof. vm_compute. auto. Qed.

Theorem unboxed_reads_examples : unboxed_reads_examples_statement.
Proof. vm_compute. repeat split. Qed.

Theorem unboxed_reads_example : unboxed_reads_example_statement.
Proof. vm_compute. repeat split. Qed.

Theorem continuation_example : continuation_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem spurious_keep_example : spurious_keep_example_statement.
Proof.
  do 10 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem alias_example : alias_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem loop_example : loop_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 40; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem nested_example : nested_example_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem nested_param_example : nested_param_example_statement.
Proof.
  do 13 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem nested_alias_example : nested_alias_example_statement.
Proof.
  do 13 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem deep_example : deep_example_statement.
Proof.
  do 13 (split; [ vm_compute; reflexivity | ]).
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem unboxed_call_example : unboxed_call_example_statement.
Proof.
  split.
  { intros f Hf. discriminate Hf. }
  do 19 (split; [ vm_compute; reflexivity | ]).
  split; [ exists 20; try exists no_reads; vm_compute; reflexivity | ].
  split; exists 20; try exists no_reads; vm_compute; reflexivity.
Qed.

Theorem unboxed_wrapper_example : unboxed_wrapper_example_statement.
Proof.
  split.
  { intros f Hf. discriminate Hf. }
  do 18 (split; [ vm_compute; reflexivity | ]).
  split; exists 30; try exists no_reads; vm_compute; reflexivity.
Qed.

(* ======================================================================== *)
(* With nothing unboxed                                                     *)
(* ======================================================================== *)

(* The rebuild with unboxing that unboxes nothing is the rebuild of
   FlambdaReaperCallingConventions.v: the solution without unboxed names
   is the solution, pointwise ([usage_nil] and the lemmas that follow,
   without extensionality); the decisions are those of the change of
   calling conventions, as [Decided] ([code_change_nil],
   [apply_change_nil], [handler_decisions_nil]); the wrappers agree
   ([wrapper_u_cc], [wrapper_cc_plain]) where the kinds and lengths of
   [arities_match] hold, which [cinv] carries for the continuations in
   scope; [rebuild_nil] by induction on the term, code bodies included. *)
Module WithoutUnboxing.

Import FlambdaFreshQuantifierSpec FlambdaPrimitiveProofs.

Definition is_decided (d : decision) : bool :=
  match d with
  | Decided _ => true
  | Unbox _ | Keep_unused _ => false
  end.

Section Without.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable first_fresh : positive.

Abbreviation solu := (solution_without_unboxed sol []).

Lemma usage_nil (n : node) : s_has_usage solu n = s_has_usage sol n.
Proof. destruct n; cbn; rewrite ?andb_true_r; reflexivity. Qed.

Lemma var_used_nil (x : variable) (kind : flambda_kind) :
  is_var_used solu x kind = is_var_used sol x kind.
Proof. destruct kind; cbn; rewrite ?andb_true_r; reflexivity. Qed.

Lemma rewrite_simple_nil (en : rebuild_env) (s : simple) :
  rewrite_simple solu en s = rewrite_simple sol en s.
Proof.
  destruct s as [x | sym | c]; cbn [rewrite_simple]; [ | | reflexivity ].
  - rewrite var_used_nil. reflexivity.
  - unfold is_symbol_used. rewrite usage_nil. reflexivity.
Qed.

Lemma rewrite_simples_nil (en : rebuild_env) (l : list simple) :
  map (rewrite_simple solu en) l = map (rewrite_simple sol en) l.
Proof. apply map_ext. intros s. apply rewrite_simple_nil. Qed.

Lemma rewrite_or_variable_nil (en : rebuild_env) (s : simple) :
  rewrite_or_variable solu en s = rewrite_or_variable sol en s.
Proof.
  destruct s as [x | sym | c]; cbn [rewrite_or_variable]; try reflexivity.
  rewrite var_used_nil. reflexivity.
Qed.

Lemma rebuild_fields_nil (en : rebuild_env) (base : node) (l : list simple) :
  ∀ i : nat, rebuild_fields solu en base i l = rebuild_fields sol en base i l.
Proof.
  induction l as [ | s l IH]; intros i; [ reflexivity | ].
  cbn [rebuild_fields]. rewrite IH, rewrite_simple_nil. reflexivity.
Qed.

Lemma rebuild_value_slots_nil (en : rebuild_env) (bound : list node)
    (l : list (value_slot * simple)) :
  rebuild_value_slots solu en bound l = rebuild_value_slots sol en bound l.
Proof.
  induction l as [ | [w s] l IH]; [ reflexivity | ].
  unfold rebuild_value_slots in *. cbn [flat_map]. rewrite IH.
  rewrite rewrite_simple_nil. reflexivity.
Qed.

Lemma rebuild_named_nil (en : rebuild_env) (x : variable) (d : named) :
  rebuild_named solu en x d = rebuild_named sol en x d.
Proof.
  destruct d as [s | p args | | | | | ]; cbn [rebuild_named];
    rewrite ?rewrite_simple_nil; try reflexivity.
  destruct p as [p | p | p | p | p]; rewrite ?rewrite_simples_nil;
    try reflexivity;
    destruct p; rewrite ?rebuild_fields_nil, ?rewrite_simples_nil;
    reflexivity.
Qed.

Lemma rebuild_set_nil (en : rebuild_env) (bound : list node) (d : named) :
  rebuild_set solu en bound d = rebuild_set sol en bound d.
Proof.
  destruct d; cbn [rebuild_set]; rewrite ?rebuild_value_slots_nil;
    reflexivity.
Qed.

Lemma rebuild_static_const_nil (en : rebuild_env) (s : symbol)
    (c : static_const) :
  rebuild_static_const solu en s c = rebuild_static_const sol en s c.
Proof.
  destruct c; cbn [rebuild_static_const];
    rewrite ?rebuild_value_slots_nil, ?rebuild_fields_nil,
      ?rewrite_or_variable_nil, ?rewrite_simples_nil; reflexivity.
Qed.

Lemma cofield_nil (in_ : node -> bool) (i : nat) :
  cofield_has_use F solu in_ i = cofield_has_use F sol in_ i.
Proof.
  unfold cofield_has_use. induction (f_parameter F) as [ | [[x j] y] l IH];
    [ reflexivity | ].
  cbn [existsb]. rewrite IH, usage_nil. reflexivity.
Qed.

Lemma arguments_used_by_call_nil (ep : closure_entry_point)
    (na : list nat) :
  ∀ cs : allocation_sources,
    arguments_used_by_call F solu ep cs na
    = arguments_used_by_call F sol ep cs na.
Proof.
  induction na as [ | n na IH]; intros cs; [ reflexivity | ].
  cbn [arguments_used_by_call]. destruct cs as [ | mem]; [ reflexivity | ].
  change (get_field_sources F solu) with (get_field_sources F sol).
  destruct (get_field_sources F sol mem (Field_call_witness ep))
    as [ | ws]; [ reflexivity | ].
  f_equal; [ apply map_ext; intros i; apply cofield_nil | ].
  destruct na; [ reflexivity | apply IH ].
Qed.

Lemma rebuild_arguments_nil (en : rebuild_env) (ap : apply_expr) :
  rebuild_arguments F solu en ap = rebuild_arguments F sol en ap.
Proof.
  unfold rebuild_arguments. destruct (ap_callee ap);
    [ | | apply rewrite_simples_nil ];
    unfold arguments_used_by_known_arity_call,
      arguments_used_by_unknown_arity_call;
    change (get_direct_sources solu) with (get_direct_sources sol);
    rewrite !arguments_used_by_call_nil; reflexivity.
Qed.

Definition decided (d : param_decision) : decision := Decided d.

Lemma decisions_u_nil (nodes : list node) (kinds : list flambda_kind) :
  decisions_u sol [] nodes kinds = map decided (decisions_of sol nodes kinds).
Proof.
  unfold decisions_u, decisions_of. rewrite map_map. apply map_ext.
  intros [n k]. reflexivity.
Qed.

Lemma code_change_nil (id : code_id) (md : code_metadata) :
  code_change_u sol cannot_change code_deps unit_code_ids [] id md
  = match
      code_calling_convention_change sol cannot_change code_deps
        unit_code_ids id md
    with
    | Not_changing_calling_convention => Not_changing_u
    | Changing_calling_convention pd rd =>
      Changing_u (map decided pd) (map decided rd)
    end.
Proof.
  unfold code_change_u, code_calling_convention_change.
  destruct (cannot_change_calling_convention cannot_change unit_code_ids id);
    [ reflexivity | ].
  destruct (lookup id code_deps); [ | reflexivity ].
  rewrite !decisions_u_nil. reflexivity.
Qed.

Lemma my_closure_nil (id : code_id) :
  my_closure_decision_u cannot_change code_deps unit_code_ids [] id = None.
Proof.
  unfold my_closure_decision_u.
  destruct (cannot_change_calling_convention cannot_change unit_code_ids id);
    [ reflexivity | ].
  destruct (lookup id code_deps); reflexivity.
Qed.

Lemma apply_my_closure_nil (ap : apply_expr) :
  apply_my_closure_u F sol cannot_change code_deps unit_code_ids codes [] ap
  = None.
Proof.
  unfold apply_my_closure_u.
  destruct (code_id_actually_called F sol ap); [ | reflexivity ].
  destruct (lookup _ codes); [ apply my_closure_nil | reflexivity ].
Qed.

Lemma apply_change_nil (ap : apply_expr) :
  apply_change_u F sol cannot_change code_deps unit_code_ids codes [] ap
  = match
      apply_calling_convention_change F sol cannot_change code_deps
        unit_code_ids codes ap
    with
    | Not_changing_calling_convention => Not_changing_u
    | Changing_calling_convention pd rd =>
      Changing_u (map decided pd) (map decided rd)
    end.
Proof.
  unfold apply_change_u, apply_calling_convention_change,
    get_calling_convention_change.
  destruct (code_id_actually_called F sol ap); [ | reflexivity ].
  destruct (lookup _ codes); [ apply code_change_nil | reflexivity ].
Qed.

Lemma get_args_with_kinds_nil (en : rebuild_env) (args : list simple) :
  ∀ ds : list param_decision,
    get_args_with_kinds_u sol [] en (map decided ds) args
    = Some (get_args_with_kinds sol en ds args).
Proof.
  induction args as [ | a args IH]; intros [ | d ds]; try reflexivity.
  cbn [map get_args_with_kinds_u]. rewrite IH.
  unfold get_args_with_kinds. cbn [combine flat_map].
  destruct d as [kind | ]; cbn [decided]; [ | reflexivity ].
  rewrite rewrite_simple_nil. reflexivity.
Qed.

Lemma group_by_sizes_map {A B : Type} (f : A -> B) (sizes : list nat) :
  ∀ l : list A,
    group_by_sizes sizes (map f l) = map (map f) (group_by_sizes sizes l).
Proof.
  induction sizes as [ | n sizes IH]; intros l; [ reflexivity | ].
  cbn [group_by_sizes map]. rewrite firstn_map, skipn_map, IH. reflexivity.
Qed.

Lemma group_by_parameter_map {A B : Type} (f : A -> B) (a : arity)
    (l : list A) :
  group_by_parameter a (map f l) = map (map f) (group_by_parameter a l).
Proof. apply group_by_sizes_map. Qed.

Lemma all_some_cons {A : Type} (o : option A) (l : list (option A)) :
  all_some (o :: l)
  = match o, all_some l with
    | Some a, Some acc => Some (a :: acc)
    | _, _ => None
    end.
Proof. reflexivity. Qed.

Lemma all_some_groups (en : rebuild_env) (l1 : list (list simple)) :
  ∀ l2 : list (list param_decision),
    all_some
      (map
         (fun '(args, decisions) =>
            get_args_with_kinds_u sol [] en decisions args)
         (combine l1 (map (map decided) l2)))
    = Some
        (map
           (fun '(args, decisions) => get_args_with_kinds sol en decisions args)
           (combine l1 l2)).
Proof.
  induction l1 as [ | a l1 IH]; intros [ | ds l2]; try reflexivity.
  cbn [map combine]. rewrite all_some_cons, IH. cbn beta iota.
  rewrite get_args_with_kinds_nil. reflexivity.
Qed.

Lemma prepend_nil {A : Type} (groups : list (list A)) :
  prepend_first [] groups = groups.
Proof. destruct groups; reflexivity. Qed.

Lemma decisions_kinds_nil (ds : list param_decision) :
  decisions_kinds (map decided ds) = kept_kinds ds.
Proof.
  induction ds as [ | [k | ] ds IH]; [ reflexivity | | ];
    cbn [map decisions_kinds flat_map]; unfold decisions_kinds in IH;
    rewrite IH; reflexivity.
Qed.

Lemma arity_of_decisions_nil (ds : list param_decision) :
  arity_of_decisions_u (map decided ds) = arity_of_decisions ds.
Proof.
  unfold arity_of_decisions_u, arity_of_decisions.
  rewrite decisions_kinds_nil. reflexivity.
Qed.

Lemma arity_of_grouped_nil (gs : list (list param_decision)) :
  arity_of_grouped_decisions_u
    (prepend_first [Decided Delete] (map (map decided) gs))
  = arity_of_grouped_decisions gs.
Proof.
  unfold arity_of_grouped_decisions_u, arity_of_grouped_decisions.
  destruct gs as [ | g gs]; [ reflexivity | ].
  cbn [map prepend_first app]. f_equal.
  - change (decisions_kinds (Decided Delete :: map decided g))
      with (decisions_kinds (map decided g)).
    rewrite decisions_kinds_nil. reflexivity.
  - rewrite map_map. apply map_ext. intros g'.
    rewrite decisions_kinds_nil. reflexivity.
Qed.

Lemma get_extra_args_nil (en : rebuild_env)
    (args : list (simple * flambda_kind)) :
  ∀ ds : list decision,
    forallb is_decided ds = true ->
    get_extra_args sol [] en ds args
    = Some (filter_kept (map is_kept_decision ds)
              (map (fun '(s, kind) => (rewrite_simple sol en s, kind)) args)).
Proof.
  induction args as [ | [s k] args IH]; intros ds Hds;
    [ destruct ds; reflexivity | ].
  destruct ds as [ | d ds].
  - cbn [get_extra_args]. rewrite (IH [] eq_refl), rewrite_simple_nil.
    reflexivity.
  - cbn [forallb] in Hds. apply andb_true_iff in Hds as [Hd Hds].
    cbn [get_extra_args]. rewrite (IH ds Hds).
    destruct d as [[kind | ] | fields | kind']; cbn in Hd; try discriminate Hd;
      cbn; rewrite ?rewrite_simple_nil; reflexivity.
Qed.

Lemma get_args_nil (en : rebuild_env) (args : list simple) :
  ∀ ds : list decision,
    forallb is_decided ds = true ->
    get_args sol [] en ds args
    = Some (filter_kept (map is_kept_decision ds)
              (map (rewrite_simple sol en) args)).
Proof.
  induction args as [ | a args IH]; intros ds Hds;
    [ destruct ds; reflexivity | ].
  destruct ds as [ | d ds].
  - cbn [get_args]. rewrite (IH [] eq_refl), rewrite_simple_nil.
    reflexivity.
  - cbn [forallb] in Hds. apply andb_true_iff in Hds as [Hd Hds].
    cbn [get_args]. rewrite (IH ds Hds).
    destruct d as [[kind | ] | fields | kind']; cbn in Hd; try discriminate Hd;
      cbn; rewrite ?rewrite_simple_nil; reflexivity.
Qed.

(* The wrappers. *)

Definition kinds_agree (rd : list param_decision) (ks : list flambda_kind)
    : bool :=
  forallb
    (fun '(d, kind) =>
       match d with
       | Keep k => kind_eqb k kind
       | Delete => true
       end)
    (combine rd ks).

Lemma freshen_params (rd : list param_decision) :
  ∀ (ks : list flambda_kind) (next : positive),
    kinds_agree rd ks = true ->
    wrapper_params (freshen_decisions (combine (map decided rd) ks) next)
    = combine
        (names_from (length (kept_kinds (map fst (combine rd ks)))) next)
        (kept_kinds (map fst (combine rd ks))).
Proof.
  induction rd as [ | [k | ] rd IH]; intros [ | kind ks] next Hk;
    try reflexivity; unfold kinds_agree in Hk; cbn [combine forallb] in Hk;
    apply andb_true_iff in Hk as [Hk0 Hk].
  - apply kind_eqb_eq in Hk0. subst kind.
    cbn [map combine freshen_decisions decided]. unfold wrapper_params.
    cbn [flat_map]. fold (wrapper_params
      (freshen_decisions (combine (map decided rd) ks) (Pos.succ next))).
    rewrite (IH ks (Pos.succ next) Hk). reflexivity.
  - cbn [map combine freshen_decisions decided]. unfold wrapper_params.
    cbn [flat_map app]. fold (wrapper_params
      (freshen_decisions (combine (map decided rd) ks) next)).
    rewrite (IH ks next Hk). reflexivity.
Qed.

Lemma wrapper_args_vars (fd : list (param_decision * flambda_kind)) :
  ∀ (mask : list bool) (next y : positive),
    Var y ∈ wrapper_args next mask fd -> (next <= y)%positive.
Proof.
  induction fd as [ | [[k | ] kind] fd IH]; intros [ | keep mask] next y Hin;
    cbn [wrapper_args] in Hin; try destruct Hin.
  - destruct keep; cbn [app] in Hin.
    + destruct Hin as [[= <-] | Hin]; [ lia | ].
      specialize (IH mask (Pos.succ next) y Hin). lia.
    + specialize (IH mask (Pos.succ next) y Hin). lia.
  - destruct keep; cbn [app] in Hin.
    + destruct Hin as [Heq | Hin]; [ discriminate Heq | ].
      exact (IH mask next y Hin).
    + exact (IH mask next y Hin).
Qed.

Lemma args_are_params_wrapper (fd : list (param_decision * flambda_kind)) :
  ∀ (mask : list bool) (next : positive),
    length fd <= length mask ->
    args_are_params (wrapper_args next mask fd)
      (names_from (length (kept_kinds (map fst fd))) next)
    = forallb (fun '(keep, (d, _)) => Bool.eqb keep (is_kept d))
        (combine mask fd).
Proof.
  induction fd as [ | [[k | ] kind] fd IH]; intros [ | keep mask] next Hlen;
    cbn [length] in Hlen; try lia; try reflexivity.
  - cbn [wrapper_args map kept_kinds flat_map fst app length names_from
         combine forallb is_kept].
    change (flat_map _ (map fst fd)) with (kept_kinds (map fst fd)).
    destruct keep; cbn [app Bool.eqb andb].
    + cbn [args_are_params]. rewrite Pos.eqb_refl, IH by lia. reflexivity.
    + destruct (wrapper_args (Pos.succ next) mask fd) as [ | [y | | ] l]
        eqn:Hw; try reflexivity.
      cbn [args_are_params].
      assert (Hy : (Pos.succ next <= y)%positive)
        by (apply (wrapper_args_vars fd mask); rewrite Hw; left;
            reflexivity).
      destruct (Pos.eqb_spec y next); [ lia | reflexivity ].
  - cbn [wrapper_args map kept_kinds flat_map fst app combine forallb
         is_kept].
    change (flat_map _ (map fst fd)) with (kept_kinds (map fst fd)).
    destruct keep; cbn [app Bool.eqb andb]; [ reflexivity | ].
    apply IH. lia.
Qed.

Lemma wrapper_args_firstn (fd : list (param_decision * flambda_kind)) :
  ∀ (l : list bool) (next : positive),
    wrapper_args next (firstn (length fd) l) fd = wrapper_args next l fd.
Proof.
  induction fd as [ | [[k | ] kind] fd IH]; intros [ | b l] next;
    try reflexivity; cbn [length firstn wrapper_args]; rewrite IH;
    reflexivity.
Qed.

Lemma combine_firstn {A : Type} (fd : list A) :
  ∀ l : list bool, combine (firstn (length fd) l) fd = combine l fd.
Proof.
  induction fd as [ | a fd IH]; intros [ | b l]; try reflexivity.
  cbn [length firstn combine]. rewrite IH. reflexivity.
Qed.

Fixpoint keeps_prefix (ads : list decision) (ks : list flambda_kind)
    : bool :=
  match ads, ks with
  | [], _ => true
  | Decided (Keep k) :: ads, k' :: ks => kind_eqb k k' && keeps_prefix ads ks
  | Decided Delete :: ads, _ :: ks => keeps_prefix ads ks
  | _, _ => false
  end.

Lemma keeps_prefix_decided (ads : list decision) :
  ∀ ks : list flambda_kind,
    keeps_prefix ads ks = true -> forallb is_decided ads = true.
Proof.
  induction ads as [ | [[k | ] | fields | k''] ads IH]; intros [ | k' ks] H;
    cbn in H |- *; try discriminate H; try reflexivity.
  - apply andb_true_iff in H as [_ H]. exact (IH ks H).
  - exact (IH ks H).
Qed.

Lemma keeps_prefix_length (ads : list decision) :
  ∀ ks : list flambda_kind,
    keeps_prefix ads ks = true -> length ads <= length ks.
Proof.
  induction ads as [ | [[k | ] | fields | k''] ads IH]; intros [ | k' ks] H;
    cbn in H |- *; try discriminate H; try lia.
  - apply andb_true_iff in H as [_ H]. specialize (IH ks H). lia.
  - specialize (IH ks H). lia.
Qed.

Lemma wrapper_args_u_nil (rd : list param_decision) :
  ∀ (ks : list flambda_kind) (ads : list decision) (next : positive)
    (m : nat),
    keeps_prefix ads ks = true -> length (combine rd ks) <= m ->
    wrapper_args_u false ads
      (freshen_decisions (combine (map decided rd) ks) next)
    = Some (wrapper_args next (map is_kept_decision ads ++ repeat true m)
              (combine rd ks)).
Proof.
  induction rd as [ | d rd IH]; intros ks ads next m Hads Hm.
  - destruct ads, m; reflexivity.
  - destruct ks as [ | kind ks];
      [ destruct ads as [ | ? [ | ]], m; reflexivity | ].
    cbn [length combine] in Hm.
    destruct ads as [ | ad ads].
    + destruct m as [ | m]; [ lia | ].
      cbn [map app repeat].
      destruct d as [k | ]; cbn [map combine freshen_decisions decided
                                 wrapper_args_u wrapper_args];
        rewrite (IH ks [] _ m eq_refl ltac:(lia)); reflexivity.
    + destruct ad as [[kind' | ] | fields | kind'']; cbn [keeps_prefix] in Hads;
        try discriminate Hads;
        [ apply andb_true_iff in Hads as [Hk Hads];
          apply kind_eqb_eq in Hk; subst kind' | ];
        cbn [map app];
        destruct d as [k | ]; cbn [map combine freshen_decisions decided
                                   wrapper_args_u wrapper_args];
          rewrite (IH ks ads _ m Hads ltac:(lia)); reflexivity.
Qed.

Lemma lookup_map_values' {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k', a) => (k', f a)) l) = option_map f (lookup k l).
Proof.
  induction l as [ | [k' a] l IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma names_from_length (n : nat) :
  ∀ x : positive, length (names_from n x) = n.
Proof. induction n as [ | n IH]; intros x; cbn; [ | rewrite IH ]; reflexivity.
Qed.

Lemma map_fst_combine' {A B : Type} (l : list A) (l' : list B) :
  length l = length l' -> map fst (combine l l') = l.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l'] H; cbn in H |- *;
    try discriminate H; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma map_snd_combine' {A B : Type} (l : list A) (l' : list B) :
  length l = length l' -> map snd (combine l l') = l'.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l'] H; cbn in H |- *;
    try discriminate H; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma firstn_length_app {A : Type} (n : nat) (l l' : list A) :
  n <= length (l ++ l') -> length (firstn n (l ++ l')) = n.
Proof. intros H. rewrite length_firstn. lia. Qed.

Lemma wrapper_u_cc (en : unbox_env) (ap : apply_expr)
    (rd : list param_decision) (ks : list flambda_kind) :
  env_conts (ue_env en)
  = map (fun '(k, ds) => (k, map is_kept_decision ds)) (ue_decisions en) ->
  (∀ ds : list decision,
     lookup (ap_continuation ap) (ue_decisions en) = Some ds ->
     keeps_prefix ds ks = true) ->
  kinds_agree rd ks = true ->
  make_apply_wrapper_u false first_fresh en ap (combine (map decided rd) ks)
  = make_apply_wrapper_cc first_fresh (ue_env en) ap (combine rd ks).
Proof.
  intros Hconts Hpre Hk. unfold make_apply_wrapper_u, make_apply_wrapper_cc,
    apply_decisions.
  rewrite Hconts, lookup_map_values'.
  set (ads := match lookup (ap_continuation ap) (ue_decisions en) with
              | Some ds => ds
              | None => []
              end).
  assert (Hads : keeps_prefix ads ks = true)
    by (unfold ads; destruct (lookup _ (ue_decisions en)) eqn:Hl;
        [ exact (Hpre _ eq_refl) | reflexivity ]).
  replace (match option_map (map is_kept_decision)
                   (lookup (ap_continuation ap) (ue_decisions en)) with
           | Some mask => mask
           | None => []
           end)
    with (map is_kept_decision ads)
    by (unfold ads; destruct (lookup _ (ue_decisions en)); reflexivity).
  set (fd := combine rd ks).
  rewrite (wrapper_args_u_nil rd ks ads first_fresh (length fd) Hads
             (le_n _)).
  rewrite (freshen_params rd ks first_fresh Hk). fold fd.
  rewrite map_fst_combine', map_snd_combine' by apply names_from_length.
  set (L := map is_kept_decision ads ++ repeat true (length fd)).
  rewrite <- (wrapper_args_firstn fd L first_fresh).
  rewrite (args_are_params_wrapper fd (firstn (length fd) L) first_fresh)
    by (unfold L; rewrite firstn_length_app; [ lia | ];
        rewrite length_app, repeat_length; lia).
  reflexivity.
Qed.

Lemma kept_kinds_keep (ks : list flambda_kind) :
  kept_kinds (map fst (combine (map Keep ks) ks)) = ks.
Proof.
  induction ks as [ | k ks IH]; [ reflexivity | ]. cbn. unfold kept_kinds in IH.
  rewrite IH. reflexivity.
Qed.

Lemma forallb_keep (ks : list flambda_kind) :
  ∀ l : list bool,
    forallb (fun '(keep, (d, _)) => Bool.eqb keep (is_kept d))
      (combine l (combine (map Keep ks) ks))
    = forallb (fun b => b) (firstn (length ks) l).
Proof.
  induction ks as [ | k ks IH]; intros [ | b l]; try reflexivity.
  cbn [map combine forallb length firstn is_kept]. rewrite IH.
  destruct b; reflexivity.
Qed.

Lemma forallb_firstn_pad (mask : list bool) :
  ∀ n : nat,
    length mask <= n ->
    forallb (fun b => b) (firstn n (mask ++ repeat true n))
    = forallb (fun b => b) mask.
Proof.
  enough (H : ∀ m n : nat,
             length mask <= n ->
             forallb (fun b => b) (firstn n (mask ++ repeat true m))
             = forallb (fun b => b) mask).
  { intros n Hn. exact (H n n Hn). }
  induction mask as [ | b mask IH]; intros m n Hn.
  - cbn [app]. clear Hn. revert n. induction m as [ | m IHm]; intros n;
      [ destruct n; reflexivity | ].
    destruct n as [ | n]; [ reflexivity | ]. cbn. apply (IHm n).
  - destruct n as [ | n]; cbn [length] in Hn; [ lia | ].
    cbn [app firstn forallb]. rewrite (IH m n) by lia. reflexivity.
Qed.

Lemma wrapper_args_keep (ks : list flambda_kind) :
  ∀ (l : list bool) (next : positive),
    length ks <= length l ->
    wrapper_args next l (combine (map Keep ks) ks)
    = filter_kept l (map Var (names_from (length ks) next)).
Proof.
  induction ks as [ | k ks IH]; intros [ | b l] next Hlen;
    cbn [length] in Hlen; try lia; try reflexivity.
  cbn [map combine wrapper_args names_from length filter_kept].
  rewrite IH by lia. destruct b; reflexivity.
Qed.

Lemma filter_kept_firstn {A : Type} (xs : list A) :
  ∀ l : list bool, filter_kept (firstn (length xs) l) xs = filter_kept l xs.
Proof.
  induction xs as [ | x xs IH]; intros [ | b l]; try reflexivity.
  cbn [length firstn filter_kept]. rewrite IH. reflexivity.
Qed.

Lemma filter_kept_pad {A : Type} (mask : list bool) :
  ∀ (m : nat) (xs : list A),
    filter_kept (mask ++ repeat true m) xs = filter_kept mask xs.
Proof.
  induction mask as [ | b mask IH]; intros m xs.
  - cbn [app]. revert xs. induction m as [ | m IHm]; intros [ | x xs];
      try reflexivity.
    cbn [repeat filter_kept]. rewrite IHm. destruct xs; reflexivity.
  - destruct xs as [ | x xs]; [ reflexivity | ].
    cbn [app filter_kept]. rewrite IH. reflexivity.
Qed.

Lemma wrapper_cc_plain (en : rebuild_env) (ap : apply_expr)
    (ks : list flambda_kind) :
  ks = unarize (ap_return_arity ap) ->
  (∀ mask : list bool,
     lookup (ap_continuation ap) (env_conts en) = Some mask ->
     length mask <= length ks) ->
  make_apply_wrapper_cc first_fresh en ap (combine (map Keep ks) ks)
  = make_apply_wrapper first_fresh en ap.
Proof.
  intros Hks Hlen. unfold make_apply_wrapper_cc, make_apply_wrapper,
    apply_decisions.
  rewrite <- Hks.
  assert (Hn : length (combine (map Keep ks) ks) = length ks)
    by (rewrite length_combine, length_map; lia).
  rewrite Hn, forallb_keep, kept_kinds_keep.
  rewrite firstn_firstn, Nat.min_id.
  destruct (lookup (ap_continuation ap) (env_conts en)) as [mask | ] eqn:Hl.
  - specialize (Hlen mask eq_refl).
    rewrite (forallb_firstn_pad mask (length ks) Hlen).
    destruct (forallb (fun b => b) mask); [ reflexivity | ].
    rewrite wrapper_args_keep
      by (rewrite firstn_length_app; [ lia | ];
          rewrite length_app, repeat_length; lia).
    set (xs := map Var (names_from (length ks) first_fresh)).
    assert (Hxs : length xs = length ks)
      by (unfold xs; rewrite length_map; apply names_from_length).
    rewrite <- Hxs, filter_kept_firstn, filter_kept_pad. reflexivity.
  - rewrite (forallb_firstn_pad [] (length ks) ltac:(cbn; lia)).
    reflexivity.
Qed.

(* Continuations and calls. *)

Record cinv (conts : finite_map (list flambda_kind)) (en : unbox_env)
    : Prop := {
  ci_drop : ue_drop_projections en = false;
  ci_conts :
    env_conts (ue_env en)
    = map (fun '(k, ds) => (k, map is_kept_decision ds)) (ue_decisions en);
  ci_prefix :
    ∀ (k : continuation) (ds : list decision),
      lookup k (ue_decisions en) = Some ds ->
      ∃ ks : list flambda_kind,
        lookup k conts = Some ks ∧ keeps_prefix ds ks = true
}.

Lemma filter_kept_nil {A : Type} (l : list A) : filter_kept [] l = l.
Proof. induction l as [ | a l IH]; cbn; [ | rewrite IH ]; reflexivity. Qed.

Lemma lookup_conts (conts : finite_map (list flambda_kind)) (en : unbox_env)
    (k : continuation) :
  cinv conts en ->
  lookup k (env_conts (ue_env en))
  = option_map (map is_kept_decision) (lookup k (ue_decisions en)).
Proof. intros Hinv. rewrite (ci_conts _ _ Hinv). apply lookup_map_values'.
Qed.

Lemma decided_of (conts : finite_map (list flambda_kind)) (en : unbox_env)
    (k : continuation) (ds : list decision) :
  cinv conts en -> lookup k (ue_decisions en) = Some ds ->
  forallb is_decided ds = true.
Proof.
  intros Hinv Hl. destruct (ci_prefix _ _ Hinv k ds Hl) as (ks & _ & Hp).
  exact (keeps_prefix_decided ds ks Hp).
Qed.

Lemma forallb_tl {A : Type} (p : A -> bool) (l : list A) :
  forallb p l = true -> forallb p (tl l) = true.
Proof. destruct l; cbn; [ reflexivity | ]. intros H.
  apply andb_true_iff in H as [_ H]. exact H. Qed.

Lemma exn_args_nil (conts : finite_map (list flambda_kind))
    (en : unbox_env) (ap : apply_expr) :
  cinv conts en ->
  rebuild_exn_extra_args_u sol [] en ap
  = Some (rebuild_exn_extra_args sol (ue_env en) ap).
Proof.
  intros Hinv. unfold rebuild_exn_extra_args_u, rebuild_exn_extra_args.
  rewrite (lookup_conts conts en _ Hinv).
  destruct (lookup (ap_exn_continuation ap) (ue_decisions en)) as [ds | ]
    eqn:Hl; cbn [option_map].
  - rewrite (get_extra_args_nil _ _ (tl ds)
               (forallb_tl _ _ (decided_of conts en _ ds Hinv Hl))).
    destruct ds; reflexivity.
  - rewrite (get_extra_args_nil _ _ [] eq_refl). cbn [map].
    rewrite filter_kept_nil. reflexivity.
Qed.

Lemma rewrite_apply_cont_nil (conts : finite_map (list flambda_kind))
    (en : unbox_env) (ac : apply_cont) :
  cinv conts en ->
  rewrite_apply_cont_u sol [] en ac = rewrite_apply_cont sol (ue_env en) ac.
Proof.
  intros Hinv. unfold rewrite_apply_cont_u, rewrite_apply_cont.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ reflexivity | ].
  rewrite (lookup_conts conts en _ Hinv).
  destruct (lookup (ac_continuation ac) (ue_decisions en)) as [ds | ]
    eqn:Hl; cbn [option_map].
  - rewrite (get_args_nil _ _ ds (decided_of conts en _ ds Hinv Hl)).
    reflexivity.
  - rewrite rewrite_simples_nil. reflexivity.
Qed.

Lemma rebuild_apply_cont_nil (conts : finite_map (list flambda_kind))
    (en : unbox_env) (ac : apply_cont) :
  cinv conts en ->
  rebuild_apply_cont_u sol [] en ac = rebuild_apply_cont sol (ue_env en) ac.
Proof.
  intros Hinv. unfold rebuild_apply_cont_u, rebuild_apply_cont.
  rewrite (rewrite_apply_cont_nil conts en ac Hinv). reflexivity.
Qed.

Lemma rebuild_switch_nil (conts : finite_map (list flambda_kind))
    (en : unbox_env) (scrutinee : simple) (arms : list (Z * apply_cont)) :
  cinv conts en ->
  rebuild_switch_u sol [] en scrutinee arms
  = rebuild_switch sol (ue_env en) scrutinee arms.
Proof.
  intros Hinv. unfold rebuild_switch_u, rebuild_switch.
  rewrite rewrite_simple_nil.
  assert (Harms :
            ∀ l : list (Z * apply_cont),
              flat_map
                (fun '(d, ac) =>
                   match rewrite_apply_cont_u sol [] en ac with
                   | Some ac => [(d, ac)]
                   | None => []
                   end)
                l
              = flat_map
                  (fun '(d, ac) =>
                     match rewrite_apply_cont sol (ue_env en) ac with
                     | Some ac => [(d, ac)]
                     | None => []
                     end)
                  l).
  { induction l as [ | [d ac] l IH]; [ reflexivity | ]. cbn [flat_map].
    rewrite IH, (rewrite_apply_cont_nil conts en ac Hinv). reflexivity. }
  rewrite Harms. reflexivity.
Qed.

Lemma keeps_prefix_trans (ds : list decision) :
  ∀ ks ks' : list flambda_kind,
    keeps_prefix ds ks = true -> is_prefix ks ks' = true ->
    keeps_prefix ds ks' = true.
Proof.
  induction ds as [ | [[k | ] | fields | k0] ds IH]; intros [ | a ks] [ | b ks']
    H1 H2; cbn in H1, H2 |- *; try discriminate; try reflexivity.
  - apply andb_true_iff in H1 as [Hk H1]. apply andb_true_iff in H2 as [Ha H2].
    apply kind_eqb_eq in Hk, Ha. subst.
    rewrite (proj2 (kind_eqb_eq b b) eq_refl).
    exact (IH ks ks' H1 H2).
  - apply andb_true_iff in H2 as [_ H2]. exact (IH ks ks' H1 H2).
Qed.

Lemma keep_map_combine (ks : list flambda_kind) :
  map (fun kind => (Decided (Keep kind), kind)) ks
  = combine (map decided (map Keep ks)) ks.
Proof. induction ks as [ | k ks IH]; cbn; [ | rewrite IH ]; reflexivity. Qed.

Lemma kinds_agree_keep (ks : list flambda_kind) :
  kinds_agree (map Keep ks) ks = true.
Proof.
  induction ks as [ | k ks IH]; [ reflexivity | ]. unfold kinds_agree in *.
  cbn. rewrite IH, (proj2 (kind_eqb_eq k k) eq_refl). reflexivity.
Qed.

Lemma kinds_agree_decisions (nodes : list node) :
  ∀ ks : list flambda_kind,
    kinds_agree (decisions_of sol nodes ks) ks = true.
Proof.
  induction nodes as [ | n nodes IH]; intros [ | k ks]; try reflexivity.
  unfold kinds_agree, decisions_of in *. cbn [combine map forallb].
  rewrite IH. destruct (is_node_used sol n k); cbn;
    [ rewrite (proj2 (kind_eqb_eq k k) eq_refl) | ]; reflexivity.
Qed.

Lemma keeps_prefix_decisions (nodes : list node) :
  ∀ ks : list flambda_kind,
    keeps_prefix (map decided (decisions_of sol nodes ks)) ks = true.
Proof.
  induction nodes as [ | n nodes IH]; intros [ | k ks]; try reflexivity.
  unfold decisions_of in *. cbn [combine map].
  destruct (is_node_used sol n k); cbn [decided keeps_prefix];
    [ rewrite (proj2 (kind_eqb_eq k k) eq_refl) | ]; apply IH.
Qed.

Definition results_of (ap : apply_expr) : option (list flambda_kind) :=
  match code_id_actually_called F sol ap with
  | Some id =>
    option_map (fun md => unarize (cm_result_arity md)) (lookup id codes)
  | None => None
  end.

Lemma changing_return (ap : apply_expr) (pd rd : list param_decision) :
  apply_calling_convention_change F sol cannot_change code_deps
    unit_code_ids codes ap
  = Changing_calling_convention pd rd ->
  ∃ (nodes : list node) (ks : list flambda_kind),
    results_of ap = Some ks ∧ rd = decisions_of sol nodes ks.
Proof.
  unfold apply_calling_convention_change, get_calling_convention_change,
    code_calling_convention_change, results_of.
  destruct (code_id_actually_called F sol ap) as [id | ]; [ | discriminate ].
  destruct (lookup id codes) as [md | ]; [ | discriminate ].
  destruct (cannot_change_calling_convention cannot_change unit_code_ids id);
    [ discriminate | ].
  destruct (lookup id code_deps) as [cd | ]; [ | discriminate ].
  intros [= _ <-]. eexists _, _. split; reflexivity.
Qed.

Lemma unboxed_of_nil (s : simple) : unboxed_of [] s = None.
Proof. destruct s; reflexivity. Qed.

Lemma rebuild_apply_nil (conts : finite_map (list flambda_kind))
    (en : unbox_env) (ap : apply_expr) :
  cinv conts en ->
  arities_match results_of code_deps conts (Apply ap) = true ->
  rebuild_apply_u false F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes [] first_fresh en ap
  = rebuild_apply_cc F sol cannot_change code_deps preserve_direct_calls
      unit_code_ids codes first_fresh (ue_env en) ap.
Proof.
  intros Hinv Hap. cbn [arities_match] in Hap.
  apply andb_true_iff in Hap as [Hk Hres].
  set (ks := unarize (ap_return_arity ap)) in *.
  assert (Hpre : ∀ ds : list decision,
             lookup (ap_continuation ap) (ue_decisions en) = Some ds ->
             keeps_prefix ds ks = true).
  { intros ds Hl. destruct (ci_prefix _ _ Hinv _ ds Hl) as (ks0 & Hl0 & Hp).
    rewrite Hl0 in Hk. exact (keeps_prefix_trans ds ks0 ks Hp Hk). }
  unfold rebuild_apply_u, rebuild_apply_cc.
  rewrite (exn_args_nil conts en ap Hinv), apply_change_nil.
  destruct (apply_calling_convention_change F sol cannot_change code_deps
              unit_code_ids codes ap) as [ | pd rd] eqn:Hch.
  - unfold rebuild_apply.
    destruct (simple_is_dead sol (ue_env en) (ap_callee ap)
              || existsb (simple_is_dead sol (ue_env en)) (ap_args ap));
      [ reflexivity | ].
    rewrite unboxed_of_nil, rebuild_arguments_nil, keep_map_combine.
    rewrite wrapper_u_cc;
      [ | exact (ci_conts _ _ Hinv) | exact Hpre | apply kinds_agree_keep ].
    apply wrapper_cc_plain; [ reflexivity | ].
    cbn [ap_continuation]. intros mask Hl.
    rewrite (lookup_conts conts en _ Hinv) in Hl.
    destruct (lookup (ap_continuation ap) (ue_decisions en)) as [ds | ]
      eqn:Hds; [ | discriminate Hl ].
    injection Hl as <-. rewrite length_map.
    exact (keeps_prefix_length ds ks (Hpre ds eq_refl)).
  - destruct (simple_is_dead sol (ue_env en) (ap_callee ap)
              || existsb (simple_is_dead sol (ue_env en)) (ap_args ap));
      [ reflexivity | ].
    rewrite apply_my_closure_nil, group_by_parameter_map, all_some_groups,
      prepend_nil, unboxed_of_nil, arity_of_decisions_nil.
    destruct (changing_return ap pd rd Hch) as (nodes & ks0 & Hr & ->).
    rewrite Hr in Hres. apply FlambdaSemanticsProofs.kinds_eqb_eq in Hres.
    subst ks0.
    apply wrapper_u_cc;
      [ exact (ci_conts _ _ Hinv) | exact Hpre | apply kinds_agree_decisions ].

Qed.

(* Handlers. *)

Definition decide_param (p : (variable * flambda_kind) * bool) : decision :=
  let '((_, kind), keep) := p in Decided (if keep then Keep kind else Delete).

Lemma handler_decisions_nil (is_exn : bool) (params : list variable)
    (md : handler_metadata) (handler : expr) :
  handler_decisions false sol [] is_exn (Handler params md handler)
  = map decide_param
      (combine (with_kinds params (hm_params_kinds md))
         (handler_mask sol is_exn (Handler params md handler))).
Proof.
  unfold handler_decisions, handler_mask. apply map_ext.
  intros [[x kind] keep]. reflexivity.
Qed.

Lemma kept_decide_params (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    map is_kept_decision (map decide_param (combine bound mask)) = mask.
Proof.
  induction bound as [ | b bound IH]; intros [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  destruct b as [x kind]. rewrite IH by lia. destruct keep; reflexivity.
Qed.

Lemma with_kinds_length' (params : list variable) :
  ∀ kinds : list flambda_kind, length (with_kinds params kinds) = length params.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds]; cbn;
    rewrite ?IH; reflexivity.
Qed.

Lemma mask_length (is_exn : bool) (params : list variable)
    (md : handler_metadata) (handler : expr) :
  length (handler_mask sol is_exn (Handler params md handler))
  = length (with_kinds params (hm_params_kinds md)).
Proof. unfold handler_mask, params_to_keep. apply length_map. Qed.

Lemma kept_handler_decisions (is_exn : bool) (h : continuation_handler) :
  map is_kept_decision (handler_decisions false sol [] is_exn h)
  = handler_mask sol is_exn h.
Proof.
  destruct h as [params md handler]. rewrite handler_decisions_nil.
  apply kept_decide_params. apply mask_length.
Qed.

Lemma with_kinds_combine' (params : list variable)
    (kinds : list flambda_kind) :
  length params = length kinds ->
  with_kinds params kinds = combine params kinds.
Proof.
  revert kinds. induction params as [ | p params IH]; intros [ | k kinds] Hlen;
    cbn in Hlen |- *; try discriminate Hlen; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma keeps_prefix_params (params : list variable) :
  ∀ (kinds : list flambda_kind) (mask : list bool),
    length params = length kinds -> length mask = length params ->
    keeps_prefix (map decide_param (combine (combine params kinds) mask))
      kinds
    = true.
Proof.
  induction params as [ | p params IH]; intros [ | k kinds] [ | keep mask]
    H1 H2; cbn in H1, H2 |- *; try lia; try reflexivity.
  destruct keep; cbn [keeps_prefix];
    [ rewrite (proj2 (kind_eqb_eq k k) eq_refl) | ]; apply IH; lia.
Qed.

Lemma keeps_prefix_handler (is_exn : bool) (h : continuation_handler) :
  length (handler_variables h) = length (handler_kinds h) ->
  keeps_prefix (handler_decisions false sol [] is_exn h) (handler_kinds h)
  = true.
Proof.
  destruct h as [params md handler]. cbn [handler_variables handler_kinds].
  intros Hlen. rewrite handler_decisions_nil.
  pose proof (mask_length is_exn params md handler) as Hm.
  rewrite with_kinds_combine' in Hm |- * by exact Hlen.
  apply keeps_prefix_params; [ exact Hlen | ].
  rewrite Hm, length_combine. lia.
Qed.

Lemma params_of_decide (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    params_of_decisions bound (map decide_param (combine bound mask))
    = filter_kept mask bound.
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  rewrite IH by lia. destruct keep; reflexivity.
Qed.

Lemma filter_kept_combine_fst {A B : Type} (l : list A) :
  ∀ (l' : list B) (mask : list bool),
    length l = length l' ->
    map fst (filter_kept mask (combine l l')) = filter_kept mask l.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  - rewrite IH by lia. reflexivity.
  - destruct keep; cbn; rewrite IH by lia; reflexivity.
Qed.

Lemma filter_kept_combine_snd {A B : Type} (l : list A) :
  ∀ (l' : list B) (mask : list bool),
    length l = length l' ->
    map snd (filter_kept mask (combine l l')) = filter_kept mask l'.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  - rewrite IH by lia. reflexivity.
  - destruct keep; cbn; rewrite IH by lia; reflexivity.
Qed.

(* Code. *)

Lemma changing_code (id : code_id) (md : code_metadata)
    (pd rd : list param_decision) :
  code_calling_convention_change sol cannot_change code_deps unit_code_ids id
    md
  = Changing_calling_convention pd rd ->
  ∃ cd : code_dep,
    lookup id code_deps = Some cd
    ∧ pd = decisions_of sol (cd_params cd) (unarize (cm_params_arity md))
    ∧ rd = decisions_of sol (cd_return cd) (unarize (cm_result_arity md)).
Proof.
  unfold code_calling_convention_change.
  destruct (cannot_change_calling_convention cannot_change unit_code_ids id);
    [ discriminate | ].
  destruct (lookup id code_deps) as [cd | ]; [ | discriminate ].
  intros [= <- <-]. exists cd. split; [ reflexivity | split; reflexivity ].
Qed.

Lemma decisions_of_length (nodes : list node) (ks : list flambda_kind) :
  length (decisions_of sol nodes ks) = Nat.min (length nodes) (length ks).
Proof. unfold decisions_of. rewrite length_map. apply length_combine. Qed.

Lemma code_params_nil (params : list variable) :
  ∀ pd : list param_decision,
    length params <= length pd ->
    map fst
      (flat_map (fun '(x, d) => decision_params x (param_decision_at [] x d))
         (combine params (map decided pd)))
    = filter_kept (map is_kept pd) params.
Proof.
  induction params as [ | p params IH]; intros [ | d pd] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  destruct d as [k | ]; cbn; rewrite IH by lia; reflexivity.
Qed.

Lemma metadata_nil (md : code_metadata) (pd rd : list param_decision) :
  changed_code_metadata_u md (Decided Delete) (map decided pd)
    (map decided rd)
  = changed_code_metadata md pd rd.
Proof.
  unfold changed_code_metadata_u, changed_code_metadata.
  rewrite group_by_parameter_map, arity_of_grouped_nil, arity_of_decisions_nil.
  reflexivity.
Qed.

(* The rebuilds. *)

Abbreviation urb en e :=
  (rebuild_expr_u false false F sol cannot_change code_deps
     preserve_direct_calls unit_code_ids codes [] first_fresh [] en e).
Abbreviation crb en e :=
  (rebuild_expr_cc F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids codes first_fresh en e).

Lemma cinv_bind (conts : finite_map (list flambda_kind)) (en : unbox_env)
    (bound : list (variable * flambda_kind)) (projections : list variable) :
  cinv conts en -> cinv conts (ubind_set en bound projections false).
Proof.
  intros [Hd Hc Hp]. split; [ reflexivity | exact Hc | exact Hp ].
Qed.

Lemma cinv_cont (conts : finite_map (list flambda_kind)) (en : unbox_env)
    (k : continuation) (ds : list decision) (ks : list flambda_kind) :
  cinv conts en -> keeps_prefix ds ks = true ->
  cinv ((k, ks) :: conts) (ubind_continuations en [(k, ds)]).
Proof.
  intros [Hd Hc Hp] Hds. split; [ reflexivity | | ].
  - cbn [ubind_continuations ue_env env_conts bind_continuations
         ue_decisions map app].
    rewrite Hc. reflexivity.
  - intros k' ds'. cbn [ubind_continuations ue_decisions app].
    rewrite !lookup_cons. destruct (Pos.eqb k' k).
    + intros [= <-]. exists ks. split; [ reflexivity | exact Hds ].
    + apply Hp.
Qed.

Lemma lookup_handlers {B : Type} (f : continuation_handler -> B)
    (k : continuation) (handlers : list (continuation * continuation_handler)) :
  lookup k (map (fun '(k, h) => (k, f h)) handlers)
  = option_map f (lookup k handlers).
Proof.
  induction handlers as [ | [k' h] l IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma cinv_rec (conts : finite_map (list flambda_kind)) (en : unbox_env)
    (handlers : list (continuation * continuation_handler)) :
  cinv conts en ->
  (∀ (k : continuation) (h : continuation_handler),
     (k, h) ∈ handlers ->
     length (handler_variables h) = length (handler_kinds h)) ->
  cinv (map (fun '(k, h) => (k, handler_kinds h)) handlers ++ conts)
    (ubind_continuations en
       (map (fun '(k, h) => (k, handler_decisions false sol [] false h))
          handlers)).
Proof.
  intros [Hd Hc Hp] Hlen. split; [ reflexivity | | ].
  - cbn [ubind_continuations ue_env env_conts bind_continuations
         ue_decisions]. rewrite Hc, map_app. reflexivity.
  - intros k ds. cbn [ubind_continuations ue_decisions].
    rewrite !lookup_app.
    rewrite (lookup_handlers (handler_decisions false sol [] false) k handlers),
      (lookup_handlers handler_kinds k handlers).
    destruct (lookup k handlers) as [h | ] eqn:Hl; cbn [option_map].
    + intros [= <-]. exists (handler_kinds h). split; [ reflexivity | ].
      apply keeps_prefix_handler. apply (Hlen k h).
      exact (lookup_in k h handlers Hl).
    + apply Hp.
Qed.

Lemma unboxed_names_nil (l : list variable) :
  existsb (fun y => is_unboxed_name [] (Node_variable y)) l = false.
Proof. induction l as [ | y l IH]; [ reflexivity | exact IH ]. Qed.

Lemma rename_projection_nil (d : named) : rename_projection [] d = d.
Proof.
  destruct d as [ | | | | | | m m' closure]; try reflexivity.
  destruct closure; reflexivity.
Qed.

Lemma rebuild_let_u_nil (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  rebuild_let_u sol [] [] en x kind d body
  = Let x kind (rebuild_named sol en x d) body.
Proof.
  unfold rebuild_let_u. rewrite rebuild_named_nil.
  destruct (field_read d) as [[f arg] | ]; [ | reflexivity ].
  rewrite unboxed_of_nil. destruct arg; reflexivity.
Qed.

Lemma static_set_repr_nil (s : symbol) (const : static_const) :
  static_set_repr [] s const = None.
Proof.
  destruct const as [ | slot closure_symbols function_decls value_slots
                    | | | | ]; try reflexivity.
  cbn [static_set_repr]. induction closure_symbols as [ | [? ?] l IH];
    [ reflexivity | exact IH ].
Qed.

Lemma rebuild_code_nil (id : code_id) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (cbody : expr) :
  (∀ (conts : finite_map (list flambda_kind)) (en : unbox_env),
     cinv conts en -> arities_match results_of code_deps conts cbody = true ->
     urb en cbody = crb (ue_env en) cbody) ->
  length params = length (unarize (cm_params_arity md)) ->
  (∀ cd : code_dep,
     lookup id code_deps = Some cd -> length params <= length (cd_params cd)) ->
  arities_match results_of code_deps [(ret, unarize (cm_result_arity md))]
    cbody
  = true ->
  rebuild_code_u false false F sol cannot_change code_deps preserve_direct_calls
    unit_code_ids codes [] first_fresh [] id (Code params mc ret exn md cbody)
  = rebuild_code_cc F sol cannot_change code_deps preserve_direct_calls
      unit_code_ids codes first_fresh id (Code params mc ret exn md cbody).
Proof.
  intros Hbody Hp Hcd Hm. cbn [rebuild_code_u rebuild_code_cc].
  rewrite code_change_nil.
  destruct (code_calling_convention_change sol cannot_change code_deps
              unit_code_ids id md) as [ | pd rd] eqn:Hc.
  - f_equal. destruct (existsb _ _); [ reflexivity | ].
    apply (Hbody [(ret, unarize (cm_result_arity md))]); [ | exact Hm ].
    split; [ reflexivity | reflexivity | ].
    intros k ds Hl. discriminate Hl.
  - rewrite my_closure_nil.
    destruct (changing_code id md pd rd Hc) as (cd & Hcdl & -> & ->).
    f_equal.
    + cbn [decision_params app]. apply code_params_nil.
      rewrite decisions_of_length. specialize (Hcd cd Hcdl). lia.
    + apply metadata_nil.
    + destruct (existsb _ _); [ reflexivity | ].
      replace (map is_kept (decisions_of sol (cd_return cd)
                              (unarize (cm_result_arity md))))
        with (map is_kept_decision
                (map decided (decisions_of sol (cd_return cd)
                                (unarize (cm_result_arity md)))))
        by (rewrite map_map; reflexivity).
      apply (Hbody [(ret, unarize (cm_result_arity md))]); [ | exact Hm ].
      split; [ reflexivity | reflexivity | ].
      intros k ds. cbn [ubind_variables ubind_variables_then ubind_set
                        ue_decisions ubind_continuations empty_unbox_env app].
      rewrite lookup_cons.
      destruct (Pos.eqb_spec k ret) as [-> | _]; [ | discriminate ].
      intros [= <-]. exists (unarize (cm_result_arity md)).
      split; [ apply lookup_head | apply keeps_prefix_decisions ].
Qed.

Lemma rebuild_nil :
  (∀ e : expr,
     ∀ (conts : finite_map (list flambda_kind)) (en : unbox_env),
       cinv conts en -> arities_match results_of code_deps conts e = true ->
       urb en e = crb (ue_env en) e)
  ∧ (∀ h : continuation_handler,
       ∀ (conts : finite_map (list flambda_kind)) (en : unbox_env)
         (dead is_exn : bool),
         cinv conts en ->
         handler_arities_match results_of code_deps conts h = true ->
         rebuild_handler_u false false F sol cannot_change code_deps
           preserve_direct_calls unit_code_ids codes [] first_fresh [] en dead
           (handler_decisions false sol [] is_exn h) h
         = rebuild_handler_cc F sol cannot_change code_deps
             preserve_direct_calls unit_code_ids codes first_fresh
             (ue_env en) dead (handler_mask sol is_exn h) h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH conts en Hinv Hm. cbn [arities_match] in Hm.
    cbn [rebuild_expr_u rebuild_expr_cc].
    destruct (is_pending_projection (ue_env en) x d).
    + rewrite (ci_drop _ _ Hinv), rename_projection_nil. f_equal.
      apply (IH conts); [ apply cinv_bind, Hinv | exact Hm ].
    + destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                     | m m' c];
        try (destruct (is_dead_var sol x kind); [ reflexivity | ];
             destruct (is_var_used sol x kind);
             [ unfold unboxed_of_node, changed_of_node; cbn [Keyed.lookup];
               rewrite rebuild_let_u_nil; f_equal;
               apply (IH conts); [ apply cinv_bind, Hinv | exact Hm ]
             | apply (IH conts); [ apply cinv_bind, Hinv | exact Hm ] ]).
      all: destruct (is_dead_var sol x kind || _); [ reflexivity | ];
        rewrite unboxed_names_nil;
        destruct (is_var_used sol x kind || _);
        [ unfold rebuild_set_u, changed_of_node; cbn [Keyed.lookup];
          rewrite rebuild_set_nil; f_equal;
          apply (IH conts); [ apply cinv_bind, Hinv | exact Hm ]
        | apply (IH conts); [ apply cinv_bind, Hinv | exact Hm ] ].
  - (* Let_static *)
    intros bindings body Hcodes IH conts en Hinv Hm.
    cbn [arities_match] in Hm. apply andb_true_iff in Hm as [Hb Hm].
    cbn [rebuild_expr_u rebuild_expr_cc]. f_equal.
    + rewrite forallb_forall in Hb. clear IH Hm.
      induction bindings as [ | b bindings IHb]; [ reflexivity | ].
      cbn [flat_map]. f_equal.
      * destruct b as [id c | id | s const].
        -- cbn [rebuild_binding_u rebuild_binding_cc].
           destruct (is_code_id_used sol id); [ | reflexivity ].
           destruct c as [params mc ret exn md cbody].
           specialize (Hb _ (or_introl eq_refl)). cbn beta iota in Hb.
           apply andb_true_iff in Hb as [Hb Hm].
           apply andb_true_iff in Hb as [Hp Hcd].
           apply Nat.eqb_eq in Hp.
           f_equal. f_equal.
           apply rebuild_code_nil; [ | exact Hp | | exact Hm ].
           ++ exact (Hcodes id (Code params mc ret exn md cbody)
                       (or_introl eq_refl)).
           ++ intros cd Hl. rewrite Hl in Hcd. apply Nat.leb_le, Hcd.
        -- reflexivity.
        -- cbn [rebuild_binding_u rebuild_binding_cc].
           rewrite static_set_repr_nil, rebuild_static_const_nil.
           reflexivity.
      * apply IHb.
        -- intros id c Hin. apply (Hcodes id c). right. exact Hin.
        -- intros x Hin. apply Hb. right. exact Hin.
    + apply (IH conts); [ apply cinv_bind, Hinv | exact Hm ].
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH conts en Hinv Hm. cbn [arities_match] in Hm.
    apply andb_true_iff in Hm as [Hh Hm].
    cbn [rebuild_expr_u rebuild_expr_cc].
    assert (Hlen : length (handler_variables h) = length (handler_kinds h))
      by (destruct h; cbn in Hh |- *; apply andb_true_iff in Hh as [Hh _];
          apply Nat.eqb_eq, Hh).
    pose proof (cinv_cont conts en k _ _ Hinv
                  (keeps_prefix_handler (handler_is_exn_handler h) h Hlen))
      as Hinv'.
    rewrite (IH _ _ Hinv' Hm). cbn [ubind_continuations ue_env map].
    rewrite kept_handler_decisions.
    destruct (among k _); [ | reflexivity ].
    f_equal. exact (IHh conts en true _ Hinv Hh).
  - (* Let_cont_recursive *)
    intros handlers body IHh IH conts en Hinv Hm.
    cbn [arities_match] in Hm. apply andb_true_iff in Hm as [Hhs Hm].
    rewrite forallb_forall in Hhs.
    assert (Hlen : ∀ k h, (k, h) ∈ handlers ->
                     length (handler_variables h) = length (handler_kinds h)).
    { intros k h Hin. specialize (Hhs _ Hin). destruct h; cbn in Hhs |- *.
      apply andb_true_iff in Hhs as [Hhs _]. apply Nat.eqb_eq, Hhs. }
    pose proof (cinv_rec conts en handlers Hinv Hlen) as Hinv'.
    cbn [rebuild_expr_u rebuild_expr_cc].
    assert (Henv :
              ue_env
                (ubind_continuations en
                   (map (fun '(k, h) =>
                           (k, handler_decisions false sol [] false h))
                      handlers))
              = bind_continuations (ue_env en)
                  (map (fun '(k, h) => (k, handler_mask sol false h))
                     handlers)).
    { cbn [ubind_continuations ue_env]. f_equal. rewrite map_map.
      apply map_ext. intros [k h]. rewrite kept_handler_decisions.
      reflexivity. }
    rewrite <- Henv. f_equal.
    + apply map_ext_in. intros [k h] Hin. f_equal.
      apply (IHh k h Hin _ _ false false Hinv'). exact (Hhs _ Hin).
    + exact (IH _ _ Hinv' Hm).
  - (* Apply *)
    intros ap conts en Hinv Hm. cbn [rebuild_expr_u rebuild_expr_cc].
    exact (rebuild_apply_nil conts en ap Hinv Hm).
  - intros ac conts en Hinv _. cbn [rebuild_expr_u rebuild_expr_cc].
    exact (rebuild_apply_cont_nil conts en ac Hinv).
  - intros scrutinee arms conts en Hinv _.
    cbn [rebuild_expr_u rebuild_expr_cc].
    exact (rebuild_switch_nil conts en scrutinee arms Hinv).
  - intros conts en _ _. reflexivity.
  - (* Handler *)
    intros params md handler IH conts en dead is_exn Hinv Hh.
    cbn [handler_arities_match] in Hh. apply andb_true_iff in Hh as [Hl Hm].
    apply Nat.eqb_eq in Hl.
    cbn [rebuild_handler_u rebuild_handler_cc].
    rewrite handler_decisions_nil.
    rewrite params_of_decide by apply mask_length.
    rewrite with_kinds_combine' by exact Hl.
    rewrite filter_kept_combine_fst, filter_kept_combine_snd by exact Hl.
    f_equal. destruct (dead && _); [ reflexivity | ].
    apply (IH conts); [ apply cinv_bind, Hinv | exact Hm ].
Qed.

End Without.

Lemma fold_prefix {A B C : Type} (g : list A * B -> C -> list A * B)
    (Hg : ∀ (acc : list A) (n : B) (c : C),
            g (acc, n) c = (acc, n)
            ∨ ∃ (y : A) (n' : B), g (acc, n) c = (acc ++ [y], n'))
    (l : list C) :
  ∀ (acc : list A) (n : B),
    ∃ extra : list A, fst (List.fold_left g l (acc, n)) = acc ++ extra.
Proof.
  induction l as [ | c l IH]; intros acc n.
  - exists []. rewrite app_nil_r. reflexivity.
  - cbn [List.fold_left].
    destruct (Hg acc n c) as [H | (y & n' & H)]; rewrite H.
    + exact (IH acc n).
    + destruct (IH (acc ++ [y]) n') as (extra & He). exists (y :: extra).
      rewrite He, <- app_assoc. reflexivity.
Qed.

Lemma fold_nil_next {A B C : Type} (g : list A * B -> C -> list A * B)
    (Hg : ∀ (acc : list A) (n : B) (c : C),
            g (acc, n) c = (acc, n)
            ∨ ∃ (y : A) (n' : B), g (acc, n) c = (acc ++ [y], n'))
    (l : list C) :
  ∀ (acc : list A) (n : B),
    fst (List.fold_left g l (acc, n)) = [] ->
    snd (List.fold_left g l (acc, n)) = n.
Proof.
  induction l as [ | c l IH]; intros acc n Hnil; [ reflexivity | ].
  cbn [List.fold_left] in Hnil |- *.
  destruct (Hg acc n c) as [H | (y & n' & H)]; rewrite H in Hnil |- *.
  - exact (IH acc n Hnil).
  - exfalso. destruct (fold_prefix g Hg l (acc ++ [y]) n') as (extra & He).
    rewrite He in Hnil. destruct acc; discriminate Hnil.
Qed.

Lemma unit_unboxed_next (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    (next : positive) :
  unit_unboxed_fields F T dominated to_unbox e = ([], next) ->
  next = FlambdaNameAbstractionSpec.fresh_name
    (FlambdaRenameApartSpec.names_of e).
Proof.
  unfold unit_unboxed_fields, unboxed_fields_of_names. intros H.
  match type of H with
  | List.fold_left ?g _ _ = _ =>
    assert (Hg : ∀ acc n c,
               g (acc, n) c = (acc, n)
               ∨ ∃ y n', g (acc, n) c = (acc ++ [y], n'))
  end.
  { intros acc n [a x]. cbn beta iota.
    destruct (_ && _); [ right | left; reflexivity ].
    match goal with
    | |- context [match ?t with pair _ _ => _ end] =>
      destruct t as [fields n']
    end.
    eexists _, _. reflexivity. }
  pose proof (fold_nil_next _ Hg dominated []
                (FlambdaNameAbstractionSpec.fresh_name
                   (FlambdaRenameApartSpec.names_of e)))
    as Hn.
  rewrite H in Hn. exact (Hn eq_refl).
Qed.

Lemma rebuild_unbox_without_unboxing (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (e : expr) :
  fst (unit_unboxed_fields F T dominated to_unbox e) = [] ->
  arities_match (call_results F T e) code_deps [] e = true ->
  rebuild_unit_unbox F T cannot_change code_deps dominated to_unbox
    preserve_direct_calls e
  = rebuild_unit_cc_from_tables F T cannot_change code_deps
      preserve_direct_calls e.
Proof.
  intros Hnil Hm.
  unfold rebuild_unit_unbox, rebuild_unit_unbox_repr,
    rebuild_unit_unbox_repr_mode, rebuild_unit_unbox_repr_modes,
    rebuild_unit_cc_from_tables, rebuild_unit_cc.
  destruct (unit_unboxed_fields F T dominated to_unbox e)
    as [unboxed next] eqn:Hu.
  cbn [fst] in Hnil. subst unboxed.
  rewrite <- (unit_unboxed_next F T dominated to_unbox e next Hu).
  change (unit_changed_representations (fun _ => false) F T dominated
            to_unbox [] e)
    with (@nil (node * closure_repr)).
  exact (proj1 (rebuild_nil F (solution_of_tables T)
                  (fun n => mem_node n cannot_change) code_deps
                  preserve_direct_calls
                  (FlambdaWellFormedSpec.static_names code_ids_bound e)
                  (bound_code_metadata e) next)
           e [] empty_unbox_env
           ltac:(split; [ reflexivity | reflexivity | discriminate ]) Hm).
Qed.

End WithoutUnboxing.

Theorem rebuild_unbox_without_unboxing :
  rebuild_unbox_without_unboxing_statement.
Proof.
  intros F T cannot_change code_deps dominated to_unbox preserve_direct_calls e.
  apply WithoutUnboxing.rebuild_unbox_without_unboxing.
Qed.

Theorem rebuild_unbox_without_unboxing_observably_local :
  rebuild_unbox_without_unboxing_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls Hlocal Hfrag Hstatic Hkinded Hwf Hone Hclosed Hlrs
    Hexact Hcc_closed Hwidgets Hnil Hm.
  rewrite (rebuild_unbox_without_unboxing (traverse_unit flag u e) T
             cannot_change (unit_code_deps flag u e) dominated to_unbox
             preserve_direct_calls e Hnil Hm).
  exact
    (FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably_local
       flag is_local u e T cannot_change preserve_direct_calls Hlocal Hfrag
       (static_constants_any_kind_arguments _ e Hstatic) Hkinded Hwf Hone
       Hclosed Hlrs Hexact Hcc_closed Hwidgets).
Qed.

Theorem rebuild_unbox_without_unboxing_observably :
  rebuild_unbox_without_unboxing_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls Hlocal Hproj Hfrag Hstatic Hkinded Hwf Hone
    Hclosed Hexact Hcc_closed Hwidgets Hnil Hm.
  exact (rebuild_unbox_without_unboxing_observably_local flag is_local u e T
           cannot_change dominated to_unbox preserve_direct_calls Hlocal Hfrag
           Hstatic Hkinded Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hproj)
           Hexact Hcc_closed Hwidgets Hnil Hm).
Qed.

Theorem without_unboxing_examples : without_unboxing_examples_statement.
Proof. vm_compute. auto. Qed.
